using System;
using System.IO;

class Program
{
    static void Main(string[] args)
    {
        if (args.Length == 0) {
            Console.WriteLine("Usage: dotnet run -- [train|export-onnx|infer-onnx] [...]");
            return;
        }
        var cmd = args[0];
        switch (cmd)
        {
            case "train":
                Trainer.RunTrain();
                break;
            case "export-onnx":
                OnnxExporter.ExportSimpleModelToOnnx();
                break;
            case "infer-onnx":
                if (args.Length < 2) { Console.WriteLine("Provide text"); return; }
                InferenceOnnx.Predict(args[1]);
                break;
            default:
                Console.WriteLine("Unknown command");
                break;
        }
    }
}