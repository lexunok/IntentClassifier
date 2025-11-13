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
