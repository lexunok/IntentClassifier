using System;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

public static class InferenceOnnx
{
    public static void Predict(string text)
    {
        var modelPath = "model.onnx";
        if (!File.Exists(modelPath)) { Console.WriteLine("model.onnx not found — run export-onnx first"); return; }

        using var session = new InferenceSession(modelPath);
        Console.WriteLine($"Loaded ONNX model: {modelPath}");
        Console.WriteLine($"Input text: {text}");
        // Токенизация и подготовка input_ids должны совпадать с моделью
        Console.WriteLine("Этот метод — шаблон. Подготовь input_ids как DenseTensor<long> и передай в сессию.");
    }
}