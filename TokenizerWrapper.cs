using System.Text.Json;
using Tokenizers.DotNet;

namespace IntentClassifier
{
    public class TokenizerWrapper(string pathOrJson) : IDisposable
    {
        private readonly Tokenizer _tokenizer = new(pathOrJson);
        private bool _disposed = false;
        public int VocabSize { get; private set; } = GetVocabSizeFromJson(pathOrJson);

        private static int GetVocabSizeFromJson(string jsonPath)
        {
            var jsonString = File.ReadAllText(jsonPath);
            using JsonDocument doc = JsonDocument.Parse(jsonString);

            if (doc.RootElement.TryGetProperty("model", out JsonElement modelElement) 
                && modelElement.TryGetProperty("vocab", out JsonElement vocabElement) 
                && vocabElement.ValueKind == JsonValueKind.Object
            )
            {
                int maxId = -1;
                foreach (JsonProperty property in vocabElement.EnumerateObject())
                    if (property.Value.TryGetInt32(out int id) && id > maxId) maxId = id;

                return maxId + 1;
            }

            return 50000;
        }

        public uint[] Encode(string text, int maxLen)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);

            uint[] enc = _tokenizer.Encode(text);
            if (enc.Length > maxLen)
            {
                var arr = new uint[maxLen];
                Array.Copy(enc, arr, maxLen);
                return arr;
            }
            return enc;
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                _tokenizer?.Dispose();
                _disposed = true;
                GC.SuppressFinalize(this);
            }
        }
    }
}