using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using TorchSharp;
using static TorchSharp.torch;

public class Sample
{
    public uint[] InputIds = Array.Empty<uint>();
    public int Label;
}

public static class DataLoader
{
    // Загружаем CSV (формат: text,label)
    public static List<Sample> LoadCsv(string path, TokenizerWrapper tokenizer, int maxLen = 64)
    {
        var lines = File.ReadAllLines(path);
        var list = new List<Sample>();

        foreach (var line in lines)
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            
            // Берём только 2 колонки: текст и метку
            var parts = line.Split(',', 2); 
            if (parts.Length < 2) continue;

            var text = parts[0].Trim();
            if (!int.TryParse(parts[1], out int label)) continue;

            var ids = tokenizer.Encode(text, maxLen);
            list.Add(new Sample { InputIds = ids, Label = label });
        }

        return list;
    }

    // Разбиваем на train / val
    public static (List<Sample> train, List<Sample> val) TrainValSplit(List<Sample> dataset, double valRatio = 0.2, int seed = 42)
    {
        var rnd = new Random(seed);
        var shuffled = dataset.OrderBy(_ => rnd.Next()).ToList();
        int valCount = (int)(dataset.Count * valRatio);

        var val = shuffled.Take(valCount).ToList();
        var train = shuffled.Skip(valCount).ToList();

        return (train, val);
    }

    // Батчи с перемешиванием
    public static IEnumerable<((Tensor input, Tensor mask) inputs, Tensor labels)> Batchify(
        List<Sample> dataset, Device device, int batchSize = 32, bool shuffle = true, int seed = 42)
    {
        var rnd = new Random(seed);
        var data = shuffle ? dataset.OrderBy(_ => rnd.Next()).ToList() : dataset;

        for (int i = 0; i < data.Count; i += batchSize)
        {
            var batch = data.Skip(i).Take(batchSize).ToArray();
            var maxLen = batch.Max(b => b.InputIds.Length);

            var inputIds = new long[batch.Length, maxLen];
            var attentionMask = new long[batch.Length, maxLen];
            var labels = new long[batch.Length];

            for (int r = 0; r < batch.Length; r++)
            {
                var row = batch[r].InputIds;
                for (int c = 0; c < maxLen; c++)
                {
                    if (c < row.Length)
                    {
                        inputIds[r, c] = row[c];
                        attentionMask[r, c] = 1;
                    }
                    else
                    {
                        inputIds[r, c] = 0;
                        attentionMask[r, c] = 0;
                    }
                }
                labels[r] = batch[r].Label;
            }

            var inputTensor = tensor(inputIds, dtype: ScalarType.Int64).to(device ?? CPU);
            var maskTensor = tensor(attentionMask, dtype: ScalarType.Int64).to(device ?? CPU);
            var labelTensor = tensor(labels, dtype: ScalarType.Int64).to(device ?? CPU);

            yield return ((inputTensor, maskTensor), labelTensor);
        }
    }
}