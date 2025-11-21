using System.Text.Json;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.optim.lr_scheduler;

namespace IntentClassifier
{

    /// <summary>
    /// Класс, отвечающий за весь процесс обучения модели.
    /// Он инкапсулирует логику загрузки данных, создания модели, самого обучения и валидации.
    /// </summary>
    public class Trainer : IDisposable
    {
        private readonly TrainingParameters _trainingParams;
        private readonly IntentClassifier _model;
        private readonly Device _device;
        private readonly List<Sample> _trainSet;
        private readonly List<Sample> _valSet;
        private readonly int _vocabSize;
        private readonly int _embDim;
        private readonly int _hiddenSize;
        private readonly int _numLayers;
        private readonly int _numLabels;

        private bool _disposed = false;
        private Adam? _optimizer;
        private CrossEntropyLoss? _lossFunc;

        private readonly JsonSerializerOptions _serializerOptions = new() { WriteIndented = true };

        /// <summary>
        /// Метод для оценки производительности модели на заданном наборе данных (обычно валидационном).
        /// </summary>
        private static (double loss, double acc) Evaluate(
            IntentClassifier model,
            List<Sample> dataset,
            Loss<Tensor, Tensor, Tensor> lossFunc,
            Device device,
            int batchSize
        )
        {
            // Переводим модель в режим оценки. В этом режиме отключаются Dropout и другие слои, специфичные для обучения.
            model.eval();
            double totalLoss = 0.0;
            int totalSamples = 0;
            int totalCorrect = 0;

            // 'no_grad()' отключает вычисление градиентов, что ускоряет процесс и экономит память.
            using (no_grad())
            {
                foreach (var (inputs, labels) in DataLoader.Batchify(dataset, batchSize: batchSize, device: device))
                {
                    var (input, mask) = inputs;

                    using Tensor logits = model.forward((input, mask));
                    using Tensor loss = lossFunc.forward(logits, labels);

                    totalLoss += loss.ToSingle() * (int)labels.shape[0];
                    totalSamples += (int)labels.shape[0];
                    totalCorrect += logits.argmax(1).eq(labels).sum().ToInt32();
                }
            }

            return totalSamples > 0
                ? (totalLoss / totalSamples, 100.0 * totalCorrect / totalSamples)
                : (0.0, 0.0);
        }

        private void CleanupTrainingResources()
        {
            _optimizer?.Dispose();
            _optimizer = null;

            _lossFunc?.Dispose();
            _lossFunc = null;

            _model?.Dispose();
        }

        /// <summary>
        /// Конструктор тренера. Выполняет всю подготовительную работу.
        /// </summary>
        public Trainer(TrainingParameters _trainingParams)
        {
            this._trainingParams = _trainingParams;
            _device = cuda.is_available() ? CUDA : CPU;
            Console.WriteLine($"Device: {_device}");

            // --- Подготовка данных ---
            if (!File.Exists(_trainingParams.DataPath))
            {
                Console.WriteLine($"{_trainingParams.DataPath} not found — creating tiny synthetic sample...");
                Directory.CreateDirectory(Path.GetDirectoryName(_trainingParams.DataPath) ?? "data");
                File.WriteAllLines(_trainingParams.DataPath, ["привет,0", "как дела,0", "купить билет,1", "где купить,1"]);
            }

            if (!File.Exists(_trainingParams.TokenizerPath)) 
                throw new FileNotFoundException($"Tokenizer not found: {_trainingParams.TokenizerPath}");

            List<Sample> all;
            using (var tok = new TokenizerWrapper(_trainingParams.TokenizerPath))
            {
                _vocabSize = tok.VocabSize;
                all = DataLoader.LoadCsv(_trainingParams.DataPath, tok, maxLen: _trainingParams.MaxLen);
            }
            if (all.Count == 0) throw new Exception("Dataset is empty!");

            // Разделяем все данные на обучающий и валидационный наборы.
            (_trainSet, _valSet) = DataLoader.TrainValSplit(all);

            Console.WriteLine($"Dataset: total={all.Count}, train={_trainSet.Count}, val={_valSet.Count}");

            // --- Определение параметров и создание модели ---
            _embDim = 256;
            _hiddenSize = 512;
            _numLayers = 2;
            _numLabels = all.Count > 0 ? all.Max(d => d.Label) + 1 : 1;
            Console.WriteLine($"Model params: vocab={_vocabSize}, embDim={_embDim}, hiddenSize={_hiddenSize}, numLayers={_numLayers}, classes={_numLabels}");

            _model = new IntentClassifier(_vocabSize, _embDim, _hiddenSize, _numLayers, _numLabels).to(_device);
        }

        ~Trainer() => Dispose();

        /// <summary>
        /// Основной метод, запускающий цикл обучения.
        /// </summary>
        public void Run()
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            // --- Инициализация компонентов для обучения ---

            // 1. Оптимизатор (Adam). Он будет обновлять веса модели.
            _optimizer = optim.Adam(_model.parameters(), lr: _trainingParams.LearningRate);

            // 2. Функция потерь (CrossEntropyLoss). Она вычисляет, насколько модель "ошиблась".
            _lossFunc = CrossEntropyLoss();

            try
            {
                // 3. Планировщик скорости обучения. Динамически изменяет learning rate для лучшей сходимости.
                LRScheduler scheduler = ReduceLROnPlateau(_optimizer, mode: "min", factor: 0.5, patience: 3, min_lr: [1e-6]);
                Console.WriteLine($"Using ReduceLROnPlateau scheduler: mode=min, factor=0.5, patience=3, min_lr=1e-6");

                Directory.CreateDirectory(_trainingParams.CheckpointsDir);
                string bestPath = Path.Combine(_trainingParams.CheckpointsDir, "model.pt");
                double bestValLoss = double.PositiveInfinity;
                int epochsNoImprove = 0;

                // Сохраняем конфигурацию модели. Это нужно для последующего инференса.
                var cfg = new ModelConfig
                {
                    VocabSize = _vocabSize,
                    EmbDim = _embDim,
                    HiddenSize = _hiddenSize,
                    NumLayers = _numLayers,
                    NumLabels = _numLabels,
                    MaxLen = _trainingParams.MaxLen
                };

                string cfgJson = JsonSerializer.Serialize(cfg, _serializerOptions);
                File.WriteAllText(Path.Combine(_trainingParams.CheckpointsDir, "model_config.json"), cfgJson);

                // --- Основной цикл обучения по эпохам ---
                for (int epoch = 1; epoch <= _trainingParams.Epochs; ++epoch)
                {
                    Console.WriteLine($"\n=== Epoch {epoch}/{_trainingParams.Epochs} ===");

                    // Переводим модель в режим обучения.
                    _model.train();
                    double epochLoss = 0.0;
                    int epochSamples = 0;
                    int epochCorrect = 0;
                    int batchIndex = 0;

                    // --- Цикл по батчам внутри одной эпохи ---
                    foreach (var (inputs, labels) in DataLoader.Batchify(_trainSet, batchSize: _trainingParams.BatchSize, device: _device))
                    {
                        var (input, mask) = inputs;

                        // --- Ключевой момент обучения (шаг обратного распространения ошибки) ---

                        // 1. Обнуляем градиенты с предыдущего шага.
                        _optimizer.zero_grad();

                        // 2. Прогоняем данные через модель (forward pass).
                        using Tensor logits = _model.forward((input, mask));

                        // 3. Считаем ошибку (loss).
                        using Tensor loss = _lossFunc.forward(logits, labels);

                        // 4. Вычисляем градиенты (как сильно каждый параметр повлиял на ошибку).
                        loss.backward();

                        // (Опционально) Обрезаем градиенты, чтобы избежать их "взрыва" и сделать обучение стабильнее.
                        nn.utils.clip_grad_norm_(_model.parameters(), 1.0);

                        // 5. Делаем шаг оптимизатора - обновляем веса модели.
                        _optimizer.step();
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
                        ++batchIndex;
                    }

                    double trainLoss = epochLoss / Math.Max(1, epochSamples);
                    double trainAcc = 100.0 * epochCorrect / Math.Max(1, epochSamples);
                    Console.WriteLine($"=> Train loss: {trainLoss:F4}, Train acc: {trainAcc:F2}%");

                    // --- Валидация в конце эпохи ---
                    var (valLoss, valAcc) = Evaluate(_model, _valSet, _lossFunc, _device, _trainingParams.BatchSize);
                    Console.WriteLine($"=> Val   loss: {valLoss:F4}, Val   acc: {valAcc:F2}%");

                    // Шаг планировщика скорости обучения. Он смотрит на valLoss и решает, нужно ли уменьшить learning rate.
                    scheduler.step(valLoss);

                    // --- Логика сохранения лучшей модели и ранней остановки ---
                    if (valLoss < bestValLoss - 1e-6)
                    {
                        bestValLoss = valLoss;
                        epochsNoImprove = 0;
                        _model.save(bestPath);
                        Console.WriteLine($"*** New best model (val loss {valLoss:F4}) saved to {bestPath}");
                    }
                    else ++epochsNoImprove;

                    if (epochsNoImprove >= _trainingParams.EarlyStoppingPatience)
                    {
                        Console.WriteLine($"Early stopping triggered (no improvement for {_trainingParams.EarlyStoppingPatience} epochs).");
                        break;
                    }
                }

                string finalPath = Path.Combine(_trainingParams.CheckpointsDir, "model.pt");
                Console.WriteLine($"Training finished. Last model saved to {finalPath}");
                Console.WriteLine($"Best model saved to {bestPath} (val loss {bestValLoss:F4})");
            }
            finally
            {
                CleanupTrainingResources();
            }
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                CleanupTrainingResources();
                _disposed = true;
                GC.SuppressFinalize(this);
            }
        }
    }
}