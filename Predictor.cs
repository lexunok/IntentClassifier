using System;
using System.IO;
using System.Text.Json;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using System.Text.Json.Serialization;

public class ModelConfig
{
    [JsonPropertyName("vocabSize")]
    public int VocabSize { get; set; }

    [JsonPropertyName("embDim")]
    public int EmbDim { get; set; }
    
    [JsonPropertyName("hiddenSize")]
    public int HiddenSize { get; set; }

    [JsonPropertyName("numLayers")]
    public int NumLayers { get; set; }

    [JsonPropertyName("numLabels")]
    public int NumLabels { get; set; }

    [JsonPropertyName("maxLen")]
    public int MaxLen { get; set; }
}

public class Predictor
{
    private readonly SimpleClassifier model;
    private readonly TokenizerWrapper tok;
    private readonly Device device;
    private readonly ModelConfig cfg;

    public Predictor(string checkpointsDir = "checkpoints", string tokenizerPath = "tokenizer.json")
    {
        device = cuda.is_available() ? CUDA : CPU;

        var modelPath = Path.Combine(checkpointsDir, "model.pt");
        var cfgPath = Path.Combine(checkpointsDir, "model_config.json");

        if (!File.Exists(modelPath)) throw new FileNotFoundException($"Model file not found: {modelPath}");
        if (!File.Exists(cfgPath)) throw new FileNotFoundException($"Model config not found: {cfgPath}");
        if (!File.Exists(tokenizerPath)) throw new FileNotFoundException($"Tokenizer file not found: {tokenizerPath}");

        // Читаем конфиг
        var cfgJson = File.ReadAllText(cfgPath);
        cfg = JsonSerializer.Deserialize<ModelConfig>(
            cfgJson,
            new JsonSerializerOptions { PropertyNameCaseInsensitive = true }
        ) ?? throw new Exception("Failed to deserialize model_config.json");

        // Создаём модель
        model = new SimpleClassifier(cfg.VocabSize, cfg.EmbDim, cfg.HiddenSize, cfg.NumLayers, cfg.NumLabels);
        model.load(modelPath);
        model.to(device);
        model.eval();

        // Токенизатор
        tok = new TokenizerWrapper(tokenizerPath);
    }

    public (int, float) Predict(string text)
    {
        int maxLen = cfg.MaxLen > 0 ? cfg.MaxLen : 64;

        // Обработка пустого текста
        if (string.IsNullOrWhiteSpace(text))
            return (0, 0.0f); // либо выбросить исключение, если нужно

        var ids = tok.Encode(text, maxLen); // int[]
        int len = Math.Min(ids.Length, maxLen);

        long[,] inputData = new long[1, maxLen];
        long[,] maskData = new long[1, maxLen];

        for (int i = 0; i < maxLen; i++)
        {
            if (i < len)
            {
                inputData[0, i] = ids[i];
                maskData[0, i] = ids[i] > 0 ? 1 : 0;
            }
            else
            {
                inputData[0, i] = 0;
                maskData[0, i] = 0;
            }
        }

        using var inputTensor = torch.tensor(inputData, dtype: ScalarType.Int64, device: device);
        using var maskTensor = torch.tensor(maskData, dtype: ScalarType.Int64, device: device);

        using var logits = model.forward((inputTensor, maskTensor));
        
        // Масштабирование по температуре для повышения уверенности
        const float temperature = 0.5f;
        using var probabilities = functional.softmax(logits / temperature, dim: 1);
        
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