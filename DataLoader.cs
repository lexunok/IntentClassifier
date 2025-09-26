using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using TorchSharp;
using static TorchSharp.torch;

public class Example
{
    public uint[] InputIds = [];
    public int Label;
}

public static class DataLoader
{
    // Simple CSV loader: text,label
    public static List<Example> LoadCsv(string path, TokenizerWrapper tokenizer, int maxLen=64)
    {
        var lines = File.ReadAllLines(path);
        var list = new List<Example>();
        foreach (var line in lines)
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            var parts = line.Split(',');
            if (parts.Length < 2) continue;
            var text = parts[0].Trim();
            if (!int.TryParse(parts[1], out int label)) continue;
            var ids = tokenizer.Encode(text, maxLen);
            list.Add(new Example { InputIds = ids, Label = label });
        }
        return list;
    }

    public static IEnumerable<(Tensor input, Tensor labels)> Batchify(List<Example> dataset, Device device, int batchSize=32)
    {
        var rnd = new Random(123);
        for (int i = 0; i < dataset.Count; i += batchSize)
        {
            var batch = dataset.Skip(i).Take(batchSize).ToArray();
            var maxLen = batch.Max(b => b.InputIds.Length);
            var data = new long[batch.Length, maxLen];
            for (int r = 0; r < batch.Length; r++)
            {
                var row = batch[r].InputIds;
                for (int c = 0; c < maxLen; c++) data[r, c] = c < row.Length ? row[c] : 0;
            }
            var labels = new long[batch.Length];
            for (int r = 0; r < batch.Length; r++) labels[r] = batch[r].Label;
            var inputTensor = tensor(data, dtype: ScalarType.Int64).to(device ?? CPU);
            var labelTensor = tensor(labels, dtype: ScalarType.Int64).to(device ?? CPU);
            yield return (inputTensor, labelTensor);
        }
    }
}