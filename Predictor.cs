using System;
using System.IO;
using System.Text.Json;
using TorchSharp;
using static TorchSharp.torch;

public class ModelConfig
{
    public int vocabSize { get; set; }
    public int embDim { get; set; }
    public int numLabels { get; set; }
    public int maxLen { get; set; }
}

public class Predictor
{
    private SimpleClassifier model;
    private TokenizerWrapper tok;
    private readonly Device device;
    private readonly ModelConfig cfg;

    public Predictor(string checkpointsDir, string tokenizerPath = "tokenizer.json")
    {
        device = cuda.is_available() ? CUDA : CPU;

        var modelPath = Path.Combine(checkpointsDir, "model.pt");
        var cfgPath = Path.Combine(checkpointsDir, "model_config.json");

        if (!File.Exists(modelPath)) throw new FileNotFoundException($"Model file not found: {modelPath}");
        if (!File.Exists(cfgPath)) throw new FileNotFoundException($"Model config not found: {cfgPath}");
        if (!File.Exists(tokenizerPath)) throw new FileNotFoundException($"Tokenizer file not found: {tokenizerPath}");

        // Читаем конфиг
        var cfgJson = File.ReadAllText(cfgPath);
        cfg = JsonSerializer.Deserialize<ModelConfig>(cfgJson);
        if (cfg == null) throw new Exception("Failed to deserialize model_config.json");

        // Создаём модель с теми же гиперпараметрами
        model = new SimpleClassifier(cfg.vocabSize, cfg.embDim, cfg.numLabels);
        // Важно: RegisterComponents() должен вызываться в конструкторе модели (см. SimpleClassifier)
        // После этого загружаем веса
        model.load(modelPath);

        // Переводим на устройство и в eval
        model.to(device);
        model.eval();

        // Токенизатор
        tok = new TokenizerWrapper(tokenizerPath);
    }

    public int Predict(string text)
    {
        int maxLen = cfg.maxLen > 0 ? cfg.maxLen : 64;
        var ids = tok.Encode(text, maxLen); // int[] длины maxLen

        // Формируем 2D массив [1, maxLen]
        long[,] data = new long[1, ids.Length];
        for (int i = 0; i < ids.Length; i++) data[0, i] = ids[i];

        using var inputTensor = torch.tensor(data, dtype: ScalarType.Int64, device: device);
        using var logits = model.forward(inputTensor); // shape [1, numLabels]

        // argmax по dim=1
        using var predTensor = logits.argmax(1);
        // Получаем значение (первый элемент)
        int predicted = predTensor.ToInt32(); // в TorchSharp есть методы ToInt32/ToSingle и т.д.

        return predicted;
    }
}