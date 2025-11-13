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
        // load prepared tokenizer.json (HuggingFace Tokenizer fast format)
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
                        // The vocab size is the number of entries in the vocab dictionary.
                        // Or, to be safer, find the max ID + 1.
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
        // Fallback or throw exception if vocab size can't be determined
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