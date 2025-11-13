using System;
using System.IO;
using System.Text.Json;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using System.Text.Json.Serialization;

/// <summary>
/// Класс, отвечающий за "инференс" - использование обученной модели для предсказания
/// намерения по новому тексту.
/// </summary>
public class Predictor
{
    private readonly IntentClassifier model;
    private readonly TokenizerWrapper tok;
    private readonly Device device;
    private readonly ModelConfig cfg;

    /// <summary>
    /// Конструктор загружает все необходимые для предсказания артефакты:
    /// - Веса обученной модели (.pt)
    /// - Конфигурацию модели (.json)
    /// - Токенизатор (.json)
    /// </summary>
    public Predictor(string checkpointsDir = "checkpoints", string tokenizerPath = "tokenizer.json")
    {
        device = cuda.is_available() ? CUDA : CPU;

        var modelPath = Path.Combine(checkpointsDir, "model.pt");
        var cfgPath = Path.Combine(checkpointsDir, "model_config.json");

        if (!File.Exists(modelPath)) throw new FileNotFoundException($"Model file not found: {modelPath}");
        if (!File.Exists(cfgPath)) throw new FileNotFoundException($"Model config not found: {cfgPath}");
        if (!File.Exists(tokenizerPath)) throw new FileNotFoundException($"Tokenizer file not found: {tokenizerPath}");

        // 1. Читаем конфигурацию модели из JSON.
        var cfgJson = File.ReadAllText(cfgPath);
        cfg = JsonSerializer.Deserialize<ModelConfig>(
            cfgJson,
            new JsonSerializerOptions { PropertyNameCaseInsensitive = true }
        ) ?? throw new Exception("Failed to deserialize model_config.json");

        // 2. Создаём экземпляр модели с той же архитектурой, что и при обучении.
        model = new IntentClassifier(cfg.VocabSize, cfg.EmbDim, cfg.HiddenSize, cfg.NumLayers, cfg.NumLabels);
        
        // 3. Загружаем в модель обученные веса.
        model.load(modelPath);
        model.to(device);
        
        // 4. Переводим модель в режим оценки (evaluation mode).
        // Это отключает Dropout и другие слои, нужные только для обучения.
        model.eval();

        // 5. Инициализируем токенизатор.
        tok = new TokenizerWrapper(tokenizerPath);
    }

    /// <summary>
    /// Выполняет предсказание для одной текстовой строки.
    /// </summary>
    /// <param name="text">Входной текст (команда пользователя).</param>
    /// <returns>Кортеж (ID намерения, Уверенность)</returns>
    public (int, float) Predict(string text)
    {
        int maxLen = cfg.MaxLen > 0 ? cfg.MaxLen : 64;

        if (string.IsNullOrWhiteSpace(text))
            return (0, 0.0f);

        // --- Шаг 1: Токенизация ---
        // Превращаем текст в массив ID токенов.
        var ids = tok.Encode(text, maxLen);
        int len = Math.Min(ids.Length, maxLen);

        // --- Шаг 2: Создание Тензоров ---
        // Создаем тензоры для ID токенов и для маски внимания,
        // которые будут переданы в модель.
        long[,] inputData = new long[1, maxLen];
        long[,] maskData = new long[1, maxLen];

        for (int i = 0; i < maxLen; i++)
        {
            if (i < len)
            {
                inputData[0, i] = ids[i];
                maskData[0, i] = 1; // 1 - настоящий токен
            }
            else
            {
                inputData[0, i] = 0; // 0 - padding-токен
                maskData[0, i] = 0;  // 0 - маскируем padding-токен
            }
        }

        using var inputTensor = torch.tensor(inputData, dtype: ScalarType.Int64, device: device);
        using var maskTensor = torch.tensor(maskData, dtype: ScalarType.Int64, device: device);

        // --- Шаг 3: Получение предсказания от модели ---
        // Прогоняем тензоры через модель и получаем на выходе логиты.
        using var logits = model.forward((inputTensor, maskTensor));
        
        // --- Шаг 4: Преобразование в вероятности ---
        // Применяем функцию Softmax к логитам, чтобы получить вероятности для каждого класса.
        // Сумма всех вероятностей будет равна 1.
        using var probabilities = functional.softmax(logits, dim: 1);
        
        // Находим класс с максимальной вероятностью.
        var (values, indices) = torch.max(probabilities, dim: 1);
        
        using (values)
        using (indices)
        {
            var intentId = indices.ToInt32();
            var confidence = values.ToSingle();
            return (intentId, confidence);
        }
    }
}
