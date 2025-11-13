using System;
using System.IO;
using System.Text.Json;
using System.Linq;
using Tokenizers.DotNet;

public class TokenizerWrapper
{
    private Tokenizer tokenizer;
    public int VocabSize { get; private set; }

    public TokenizerWrapper(string pathOrJson)
    {
        tokenizer = new Tokenizer(pathOrJson);
        VocabSize = GetVocabSizeFromJson(pathOrJson);
    }

    private int GetVocabSizeFromJson(string jsonPath)
    {
        var jsonString = File.ReadAllText(jsonPath);
        using (JsonDocument doc = JsonDocument.Parse(jsonString))
        {
            JsonElement root = doc.RootElement;
            if (root.TryGetProperty("model", out JsonElement modelElement))
            {
                if (modelElement.TryGetProperty("vocab", out JsonElement vocabElement))
                {
                    if (vocabElement.ValueKind == JsonValueKind.Object)
                    {
                        int maxId = -1;
                        foreach (JsonProperty property in vocabElement.EnumerateObject())
                        {
                            if (property.Value.TryGetInt32(out int id))
                            {
                                if (id > maxId)
                                {
                                    maxId = id;
                                }
                            }
                        }
                        return maxId + 1;
                    }
                }
            }
        }
        return 50000;
    }

    public uint[] Encode(string text, int maxLen)
    {
        var enc = tokenizer.Encode(text);
        if (enc.Length > maxLen)
        {
            var arr = new uint[maxLen];
            Array.Copy(enc, arr, maxLen);
            return arr;
        }
        return enc;
    }
}
