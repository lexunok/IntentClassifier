namespace IntentClassifier
{
    public record TrainingParameters(
        string DataPath = "data/train.csv",
        string TokenizerPath = "tokenizer.json",
        string CheckpointsDir = "checkpoints",
        int MaxLen = 64,
        int BatchSize = 64,
        int Epochs = 20,
        double LearningRate = 0.0005,
        int EarlyStoppingPatience = 7
    );
}