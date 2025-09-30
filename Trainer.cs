using System;
using System.IO;
using System.Linq;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using System.Text.Json;
public static class Trainer
{
    public static void RunTrain()
    {
        Console.WriteLine("Starting training...");
        var device = cuda.is_available() ? CUDA : CPU;
        Console.WriteLine($"Device: {device}");

        // Готовим токенизатор и датасет
        var tokenizerPath = Path.Combine("..","tokenizer.json");
        if (!File.Exists("data/train.csv")) {
            Console.WriteLine("data/train.csv not found — creating tiny synthetic sample...");
            Directory.CreateDirectory("data");
            File.WriteAllLines("data/train.csv", ["привет,0","как дела,0","купить билет,1","где купить,1"]);
        }
        var tok = new TokenizerWrapper("tokenizer.json");
        var dataset = DataLoader.LoadCsv("data/train.csv", tok, maxLen:64);

        int vocabSize = tok.VocabSize;
        int embDim = 128;
        int numLabels = dataset.Select(d => d.Label).Distinct().Count();

        var model = new SimpleClassifier(vocabSize, embDim, numLabels).to(device);
        var opt = optim.Adam(model.parameters(), lr: 1e-3);
        var lossFunc = CrossEntropyLoss();
        
        // Совершаем эпохи тренировки
        for (int epoch = 1; epoch <= 5; epoch++)
        {
            Console.WriteLine($"Epoch {epoch}");
            int step = 0;
            foreach (var (input, labels) in DataLoader.Batchify(dataset, batchSize: 2, device: device))
            {
                model.train();
                opt.zero_grad();
                var logits = model.forward(input);
                var loss = lossFunc.forward(logits, labels);
                loss.backward();
                opt.step();
                if (step % 10 == 0) Console.WriteLine($"step {step} loss {loss.ToSingle():F4}");
                step++;
            }
        }
        
        // Сохраняем модельку
        Directory.CreateDirectory("checkpoints");
        var savePath = Path.Combine("checkpoints", "model.pt");
        model.save(savePath);
        Console.WriteLine($"✅ Saved checkpoint to {savePath}");

        // Сохраняем конфиг модели рядом с весами
        var cfg = new {
            vocabSize,
            embDim,
            numLabels,
            maxLen = 64
        };
        var cfgJson = JsonSerializer.Serialize(cfg, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(Path.Combine("checkpoints", "model_config.json"), cfgJson);

    }
}