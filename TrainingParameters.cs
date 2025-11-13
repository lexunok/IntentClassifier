public class TrainingParameters
{
    public string DataPath { get; set; } = "data/train.csv";
    public string TokenizerPath { get; set; } = "tokenizer.json";
    public string CheckpointsDir { get; set; } = "checkpoints";
    public int MaxLen { get; set; } = 64;
    public int BatchSize { get; set; } = 64;
    public int Epochs { get; set; } = 20;
    public double LearningRate { get; set; } = 0.0005;
    public int EarlyStoppingPatience { get; set; } = 7;
}
