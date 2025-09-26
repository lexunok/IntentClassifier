using System;
using Tokenizers.DotNet;
public class TokenizerWrapper
{
    private Tokenizer tokenizer;
    public int VocabSize = 50000;

    public TokenizerWrapper(string pathOrJson)
    {
        // load prepared tokenizer.json (HuggingFace Tokenizer fast format)
        tokenizer = new Tokenizer(pathOrJson);
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