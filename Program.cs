namespace IntentClassifier
{
    class Program
    {
        private static readonly Dictionary<string, Action<string[]>> Commands = new(StringComparer.OrdinalIgnoreCase)
        {
            ["train"] = Train,
            ["infer"] = Infer
        };

        private static void Train(string[] args)
        {
            using var trainer = new Trainer(new TrainingParameters());
            trainer.Run();
        }

        private static void Infer(string[] args)
        {
            if (args.Length < 2)
            {
                Console.WriteLine("Usage for infer: dotnet run -- infer \"text\"");
                return;
            }

            using var predictor = new Predictor();
            var (intentId, confidence) = predictor.Predict(args[1]);
            Console.WriteLine($"Текст: \"{args[1]}\" → intentId: {intentId} (Уверенность: {confidence:P2})");
        }

        static void Main(string[] args)
        {
            if (args.Length == 0)
            {
                Console.WriteLine("Usage: dotnet run -- [train|infer] [...]");
                return;
            }

            if (Commands.TryGetValue(args[0], out var command)) command(args);
            else Console.WriteLine("Unknown command");
        }
    }
}