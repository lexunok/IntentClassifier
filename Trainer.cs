using System;
using System.IO;
using System.Linq;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using System.Text.Json;

/// <summary>
/// Класс, отвечающий за весь процесс обучения модели.
/// Он инкапсулирует логику загрузки данных, создания модели, самого обучения и валидации.
/// </summary>
public class Trainer
{
    private readonly TrainingParameters trainingParams;
    private readonly IntentClassifier model;
    private readonly torch.Device device;
    private readonly List<Sample> trainSet;
    private readonly List<Sample> valSet;
    private readonly DataLoader dataLoader;
    private readonly int vocabSize;
    private readonly int embDim;
    private readonly int hiddenSize;
    private readonly int numLayers;
    private readonly int numLabels;

    /// <summary>
    /// Конструктор тренера. Выполняет всю подготовительную работу.
    /// </summary>
    public Trainer(TrainingParameters trainingParams)
    {
        this.trainingParams = trainingParams;
        this.device = cuda.is_available() ? CUDA : CPU;
        Console.WriteLine($"Device: {device}");

        this.dataLoader = new DataLoader();

        // --- Подготовка данных ---
        if (!File.Exists(trainingParams.DataPath))
        {
            Console.WriteLine($"{trainingParams.DataPath} not found — creating tiny synthetic sample...");
            Directory.CreateDirectory(Path.GetDirectoryName(trainingParams.DataPath) ?? "data");
            File.WriteAllLines(trainingParams.DataPath, new[] { "привет,0", "как дела,0", "купить билет,1", "где купить,1" });
        }

        if (!File.Exists(trainingParams.TokenizerPath)) throw new FileNotFoundException($"Tokenizer not found: {trainingParams.TokenizerPath}");

        var tok = new TokenizerWrapper(trainingParams.TokenizerPath);
        var all = dataLoader.LoadCsv(trainingParams.DataPath, tok, maxLen: trainingParams.MaxLen);
        if (all.Count == 0) throw new Exception("Dataset is empty!");

        // Разделяем все данные на обучающий и валидационный наборы.
        (this.trainSet, this.valSet) = dataLoader.TrainValSplit(all);
        
        Console.WriteLine($"Dataset: total={all.Count}, train={trainSet.Count}, val={valSet.Count}");

        // --- Определение параметров и создание модели ---
        this.vocabSize = tok.VocabSize;
        this.embDim = 256;
        this.hiddenSize = 512;
        this.numLayers = 2;
        this.numLabels = all.Count > 0 ? all.Max(d => d.Label) + 1 : 1;
        Console.WriteLine($"Model params: vocab={vocabSize}, embDim={embDim}, hiddenSize={hiddenSize}, numLayers={numLayers}, classes={numLabels}");

        this.model = new IntentClassifier(vocabSize, embDim, hiddenSize, numLayers, numLabels).to(device);
    }

    /// <summary>
    /// Основной метод, запускающий цикл обучения.
    /// </summary>
    public void Run()
    {
        // --- Инициализация компонентов для обучения ---

        // 1. Оптимизатор (Adam). Он будет обновлять веса модели.
        var opt = optim.Adam(model.parameters(), lr: trainingParams.LearningRate);
        
        // 2. Функция потерь (CrossEntropyLoss). Она вычисляет, насколько модель "ошиблась".
        var lossFunc = CrossEntropyLoss();

        // 3. Планировщик скорости обучения. Динамически изменяет learning rate для лучшей сходимости.
        var scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode: "min", factor: 0.5, patience: 3, min_lr: new double[] { 1e-6 });
        Console.WriteLine($"Using ReduceLROnPlateau scheduler: mode=min, factor=0.5, patience=3, min_lr=1e-6");

        Directory.CreateDirectory(trainingParams.CheckpointsDir);
        var bestPath = Path.Combine(trainingParams.CheckpointsDir, "model.pt");
        double bestValLoss = double.PositiveInfinity;
        int epochsNoImprove = 0;

        // Сохраняем конфигурацию модели. Это нужно для последующего инференса.
        var cfg = new ModelConfig { VocabSize = vocabSize, EmbDim = embDim, HiddenSize = hiddenSize, NumLayers = numLayers, NumLabels = numLabels, MaxLen = trainingParams.MaxLen };
        var cfgJson = JsonSerializer.Serialize(cfg, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(Path.Combine(trainingParams.CheckpointsDir, "model_config.json"), cfgJson);

        // --- Основной цикл обучения по эпохам ---
        for (int epoch = 1; epoch <= trainingParams.Epochs; epoch++)
        {
            Console.WriteLine($"\n=== Epoch {epoch}/{trainingParams.Epochs} ===");
            
            // Переводим модель в режим обучения.
            model.train();
            double epochLoss = 0.0;
            int epochSamples = 0;
            int epochCorrect = 0;
            int batchIndex = 0;

            // --- Цикл по батчам внутри одной эпохи ---
            foreach (var (inputs, labels) in dataLoader.Batchify(trainSet, batchSize: trainingParams.BatchSize, device: device))
            {
                var (input, mask) = inputs;

                // --- Ключевой момент обучения (шаг обратного распространения ошибки) ---
                
                // 1. Обнуляем градиенты с предыдущего шага.
                opt.zero_grad();

                // 2. Прогоняем данные через модель (forward pass).
                using var logits = model.forward((input, mask));
                
                // 3. Считаем ошибку (loss).
                using var loss = lossFunc.forward(logits, labels);
                
                // 4. Вычисляем градиенты (как сильно каждый параметр повлиял на ошибку).
                loss.backward();

                // (Опционально) Обрезаем градиенты, чтобы избежать их "взрыва" и сделать обучение стабильнее.
                nn.utils.clip_grad_norm_(model.parameters(), 1.0);

                // 5. Делаем шаг оптимизатора - обновляем веса модели.
                opt.step();
                // --- Конец шага обучения ---

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

            // --- Валидация в конце эпохи ---
            var (valLoss, valAcc) = Evaluate(model, valSet, lossFunc, device, trainingParams.BatchSize);
            Console.WriteLine($"=> Val   loss: {valLoss:F4}, Val   acc: {valAcc:F2}%");
            
            // Шаг планировщика скорости обучения. Он смотрит на valLoss и решает, нужно ли уменьшить learning rate.
            scheduler.step(valLoss);

            // --- Логика сохранения лучшей модели и ранней остановки ---
            if (valLoss < bestValLoss - 1e-6)
            {
                bestValLoss = valLoss;
                epochsNoImprove = 0;
                model.save(bestPath);
                Console.WriteLine($"*** New best model (val loss {valLoss:F4}) saved to {bestPath}");
            }
            else
            {
                epochsNoImprove++;
            }

            if (epochsNoImprove >= trainingParams.EarlyStoppingPatience)
            {
                Console.WriteLine($"Early stopping triggered (no improvement for {trainingParams.EarlyStoppingPatience} epochs).");
                break;
            }
        }

        var finalPath = Path.Combine(trainingParams.CheckpointsDir, "model.pt");
        Console.WriteLine($"Training finished. Last model saved to {finalPath}");
        Console.WriteLine($"Best model saved to {bestPath} (val loss {bestValLoss:F4})");
    }
    
    /// <summary>
    /// Метод для оценки производительности модели на заданном наборе данных (обычно валидационном).
    /// </summary>
    private (double loss, double acc) Evaluate(
        IntentClassifier model,
        List<Sample> dataset,
        Loss<Tensor, Tensor, Tensor> lossFunc,
        Device device,
        int batchSize)
    {
        // Переводим модель в режим оценки. В этом режиме отключаются Dropout и другие слои, специфичные для обучения.
        model.eval();
        double totalLoss = 0.0;
        int totalSamples = 0;
        int totalCorrect = 0;

        // 'no_grad()' отключает вычисление градиентов, что ускоряет процесс и экономит память.
        using (no_grad())
        {
            foreach (var (inputs, labels) in dataLoader.Batchify(dataset, batchSize: batchSize, device: device))
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
