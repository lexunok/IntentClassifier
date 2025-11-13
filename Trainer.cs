using System;
using System.IO;
using System.Linq;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using System.Text.Json;
public static class Trainer
{
    private const int DEFAULT_MAXLEN = 64;
    private const int DEFAULT_BATCH = 64;
    private const int DEFAULT_EPOCHS = 20;
    private const double DEFAULT_LR = 0.0005;

    private const int LR_STEP_SIZE = 5;
    private const double LR_GAMMA = 0.5;
    private const int EARLY_STOPPING_PATIENCE = 7;

    public static void RunTrain(
        string dataPath = "data/train.csv",
        string tokenizerPath = "tokenizer.json",
        string checkpointsDir = "checkpoints",
        int maxLen = DEFAULT_MAXLEN,
        int batchSize = DEFAULT_BATCH,
        int epochs = DEFAULT_EPOCHS,
        double lr = DEFAULT_LR)
    {
        Console.WriteLine("Starting training...");
        var device = cuda.is_available() ? CUDA : CPU;
        Console.WriteLine($"Device: {device}");

        if (!File.Exists(dataPath))
        {
            Console.WriteLine("data/train.csv not found — creating tiny synthetic sample...");
            Directory.CreateDirectory(Path.GetDirectoryName(dataPath) ?? "data");
            File.WriteAllLines(dataPath, new[] { "привет,0", "как дела,0", "купить билет,1", "где купить,1" });
        }

        if (!File.Exists(tokenizerPath)) throw new FileNotFoundException($"Tokenizer not found: {tokenizerPath}");

        var tok = new TokenizerWrapper(tokenizerPath);
        var all = DataLoader.LoadCsv(dataPath, tok, maxLen: maxLen);
        if (all.Count == 0) throw new Exception("Dataset is empty!");

        var rnd = new Random(42);
        var shuffled = all.OrderBy(_ => rnd.Next()).ToList();
        int splitIndex = (int)(shuffled.Count * 0.8);
        var trainSet = shuffled.Take(splitIndex).ToList();
        var valSet = shuffled.Skip(splitIndex).ToList();

        Console.WriteLine($"Dataset: total={all.Count}, train={trainSet.Count}, val={valSet.Count}");

        int vocabSize = tok.VocabSize;
        int embDim = 256;
        int hiddenSize = 512;
        int numLayers = 2;
        int numLabels = all.Count > 0 ? all.Max(d => d.Label) + 1 : 1; // Динамическое определение numLabels
        Console.WriteLine($"Model params: vocab={vocabSize}, embDim={embDim}, hiddenSize={hiddenSize}, numLayers={numLayers}, classes={numLabels}");

        var model = new SimpleClassifier(vocabSize, embDim, hiddenSize, numLayers, numLabels).to(device);
        var opt = optim.Adam(model.parameters(), lr: lr);
        var lossFunc = CrossEntropyLoss();

        // Scheduler
        optim.lr_scheduler.LRScheduler? scheduler = null;
        try
        {
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode: "min", factor: 0.5, patience: 3, min_lr: new double[] { 1e-6 });
            Console.WriteLine($"Using ReduceLROnPlateau scheduler: mode=min, factor=0.5, patience=3, min_lr=1e-6");
        }
        catch { scheduler = null; }

        Directory.CreateDirectory(checkpointsDir);
        var bestPath = Path.Combine(checkpointsDir, "model.pt");
        double bestValLoss = double.PositiveInfinity;
        int epochsNoImprove = 0;

        // Сохраняем config заранее
        var cfg = new { vocabSize, embDim, hiddenSize, numLayers, numLabels, maxLen };
        var cfgJson = JsonSerializer.Serialize(cfg, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(Path.Combine(checkpointsDir, "model_config.json"), cfgJson);

        for (int epoch = 1; epoch <= epochs; epoch++)
        {
            Console.WriteLine($"\n=== Epoch {epoch}/{epochs} ===");
            trainSet = trainSet.OrderBy(_ => rnd.Next()).ToList();

            model.train();
            double epochLoss = 0.0;
            int epochSamples = 0;
            int epochCorrect = 0;
            int batchIndex = 0;

            foreach (var (inputs, labels) in DataLoader.Batchify(trainSet, batchSize: batchSize, device: device))
            {
                var (input, mask) = inputs;

                opt.zero_grad();

                using var logits = model.forward((input, mask));
                using var loss = lossFunc.forward(logits, labels);
                loss.backward();

                // gradient clipping
                nn.utils.clip_grad_norm_(model.parameters(), 1.0);

                opt.step();

                epochLoss += loss.ToSingle() * (int)labels.shape[0];
                epochSamples += (int)labels.shape[0];
                epochCorrect += logits.argmax(1).eq(labels).sum().ToInt32();

                if (batchIndex % 10 == 0)
                {
                    double currLoss = epochLoss / Math.Max(1, epochSamples);
                    double currAcc = 100.0 * epochCorrect / Math.Max(1, epochSamples);
                    Console.WriteLine($"  [batch {batchIndex}] loss={currLoss:F4} acc={currAcc:F2}%");
                }
                batchIndex++;
            }

            double trainLoss = epochLoss / Math.Max(1, epochSamples);
            double trainAcc = 100.0 * epochCorrect / Math.Max(1, epochSamples);
            Console.WriteLine($"=> Train loss: {trainLoss:F4}, Train acc: {trainAcc:F2}%");

            var (valLoss, valAcc) = Evaluate(model, valSet, lossFunc, device, batchSize);
            Console.WriteLine($"=> Val   loss: {valLoss:F4}, Val   acc: {valAcc:F2}%");
            
            try { scheduler?.step(valLoss); } catch { }

            if (valLoss < bestValLoss - 1e-6)
            {
                bestValLoss = valLoss;
                epochsNoImprove = 0;
                model.save(bestPath);
                Console.WriteLine($"*** New best model (val loss {valLoss:F4}) saved to {bestPath}");
            }
            else epochsNoImprove++;

            if (epochsNoImprove >= EARLY_STOPPING_PATIENCE)
            {
                Console.WriteLine($"Early stopping triggered (no improvement for {EARLY_STOPPING_PATIENCE} epochs).");
                break;
            }
        }

        // final save
        var finalPath = Path.Combine(checkpointsDir, "model.pt");
        model.save(finalPath);
        Console.WriteLine($"Training finished. Last model saved to {finalPath}");
        Console.WriteLine($"Best model saved to {bestPath} (val loss {bestValLoss:F4})");
    }

    private static (double loss, double acc) Evaluate(
        SimpleClassifier model,
        List<Sample> dataset,
        Loss<Tensor, Tensor, Tensor> lossFunc,
        Device device,
        int batchSize)
    {
        model.eval();
        double totalLoss = 0.0;
        int totalSamples = 0;
        int totalCorrect = 0;

        using (no_grad())
        {
            foreach (var (inputs, labels) in DataLoader.Batchify(dataset, batchSize: batchSize, device: device))
            {
                var (input, mask) = inputs;

                using var logits = model.forward((input, mask));
                using var loss = lossFunc.forward(logits, labels);

                totalLoss += loss.ToSingle() * (int)labels.shape[0];
                totalSamples += (int)labels.shape[0];
                totalCorrect += logits.argmax(1).eq(labels).sum().ToInt32();
            }
        }

        double avgLoss = totalSamples > 0 ? totalLoss / totalSamples : 0.0;
        double accPercent = totalSamples > 0 ? 100.0 * totalCorrect / totalSamples : 0.0;
        return (avgLoss, accPercent);
    }
}
