using System;
using System.IO;

class Program
{
    static void Main(string[] args)
    {
        if (args.Length == 0) {
            Console.WriteLine("Usage: dotnet run -- [train|infer] [...]");
            return;
        }
        var cmd = args[0];
        switch (cmd)
        {
            case "train":
                Trainer.RunTrain();
                break;
            case "infer":
                var predictor = new Predictor();
                var (intentId, confidence) = predictor.Predict(args[1]);
                Console.WriteLine($"Текст: \"{args[1]}\" → intentId: {intentId} (Уверенность: {confidence:P2})");
                break;
            default:
                Console.WriteLine("Unknown command");
                break;
        }
    }
}