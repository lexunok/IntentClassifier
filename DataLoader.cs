using static TorchSharp.torch;

namespace IntentClassifier
{
    /// <summary>
    /// Класс, отвечающий за загрузку и подготовку данных для модели.
    /// Реализует всю логику, описанную в THEORY.md, "Этап 1: Данные и Токенизация".
    /// </summary>
    public static class DataLoader
    {
        /// <summary>
        /// Загружает данные из CSV-файла (формат: text,label).
        /// </summary>
        /// <param name="path">Путь к файлу.</param>
        /// <param name="tokenizer">Экземпляр токенизатора.</param>
        /// <param name="maxLen">Максимальная длина последовательности.</param>
        /// <returns>Список объектов Sample.</returns>
        public static List<Sample> LoadCsv(string path, TokenizerWrapper tokenizer, int maxLen = 64)
        {
            string[] lines = File.ReadAllLines(path);
            List<Sample> list = [];

            foreach (var line in lines)
            {
                if (string.IsNullOrWhiteSpace(line)) continue;

                string[] parts = line.Split(',', 2);
                if (parts.Length < 2) continue;

                string text = parts[0].Trim();
                if (!int.TryParse(parts[1], out int label)) continue;

                // Токенизируем текст и получаем массив ID токенов.
                uint[] ids = tokenizer.Encode(text, maxLen);
                list.Add(new Sample { InputIds = ids, Label = label });
            }

            return list;
        }

        /// <summary>
        /// Разделяет датасет на обучающий и валидационный наборы.
        /// Валидационный набор нужен для объективной оценки качества модели во время обучения.
        /// </summary>
        public static (List<Sample> train, List<Sample> val) TrainValSplit(List<Sample> dataset, double valRatio = 0.2, int seed = 42)
        {
            var rnd = new Random(seed);
            var shuffled = dataset.OrderBy(_ => rnd.Next()).ToList();
            int valCount = (int)(dataset.Count * valRatio);

            return (shuffled.Skip(valCount).ToList(), shuffled.Take(valCount).ToList());
        }

        /// <summary>
        /// Главный метод подготовки данных.
        /// Разбивает датасет на "батчи" (пачки) и превращает их в тензоры, готовые для подачи в модель.
        /// Здесь реализуется логика паддинга (padding) и создания маски внимания (attention mask).
        /// </summary>
        public static IEnumerable<((Tensor input, Tensor mask) inputs, Tensor labels)> Batchify(
            List<Sample> dataset, Device device, int batchSize = 32, bool shuffle = true, int seed = 42)
        {
            var rnd = new Random(seed);
            List<Sample> data = shuffle ? [.. dataset.OrderBy(_ => rnd.Next())] : dataset;

            for (int i = 0; i < data.Count; i += batchSize)
            {
                var batch = data.Skip(i).Take(batchSize).ToArray();
                // Находим максимальную длину последовательности в текущем батче.
                var maxLen = batch.Max(b => b.InputIds.Length);

                // Создаем массивы для данных этого батча.
                var inputIds = new long[batch.Length, maxLen];
                var attentionMask = new long[batch.Length, maxLen];
                var labels = new long[batch.Length];

                for (int r = 0; r < batch.Length; ++r)
                {
                    uint[] row = batch[r].InputIds;
                    for (int c = 0; c < maxLen; ++c)
                    {
                        if (c < row.Length)
                        {
                            // Заполняем настоящими ID токенов.
                            inputIds[r, c] = row[c];
                            // Ставим 1 в маске - это настоящий токен.
                            attentionMask[r, c] = 1;
                        }
                        else
                        {
                            // --- Padding (Выравнивание) ---
                            // Добиваем последовательность нулями до максимальной длины в батче.
                            inputIds[r, c] = 0; // 0 - это ID padding-токена.
                                                // --- Attention Mask (Маска внимания) ---
                                                // Ставим 0 в маске - это padding-токен, на него модель не будет обращать внимание.
                            attentionMask[r, c] = 0;
                        }
                    }
                    labels[r] = batch[r].Label;
                }

                // Превращаем обычные C# массивы в тензоры PyTorch и отправляем их на нужное устройство (CPU или GPU).
                Tensor inputTensor = tensor(inputIds, dtype: ScalarType.Int64).to(device ?? CPU);
                Tensor maskTensor = tensor(attentionMask, dtype: ScalarType.Int64).to(device ?? CPU);
                Tensor labelTensor = tensor(labels, dtype: ScalarType.Int64).to(device ?? CPU);

                yield return ((inputTensor, maskTensor), labelTensor);
            }
        }
    }
}