using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace IntentClassifier
{

    /// <summary>
    /// Определяет архитектуру нашей нейронной сети для классификации текста.
    /// Эта модель принимает на вход последовательность токенов (ID слов) и маску
    /// и возвращает "логиты" - сырые оценки для каждого класса намерений.
    /// </summary>
    public class IntentClassifier : Module<(Tensor input, Tensor mask), Tensor>
    {
        // --- Слои Нейронной Сети ---

        // 1. Слой эмбеддингов (Embedding Layer)
        // Превращает ID слов (токены) в плотные векторы (эмбеддинги).
        // Это "умный словарь", где похожие по смыслу слова имеют похожие векторы.
        // Подробнее см. в THEORY.md, "Этап 2: Архитектура нейросети".
        private readonly Module<Tensor, Tensor> _embedding;

        // 2. Слой LSTM (Long Short-Term Memory)
        // Обрабатывает последовательность векторов (эмбеддингов), улавливая контекст и порядок слов.
        // 'bidirectional: true' означает, что LSTM читает предложение в двух направлениях (слева направо и справа налево) для лучшего понимания.
        // 'numLayers' позволяет делать сеть глубже, создавая иерархию признаков.
        private readonly LSTM _lstm;

        // 4. Полносвязные (линейные) слои
        // Финальные слои, которые принимают обработанную информацию и принимают решение о классификации.
        private readonly Module<Tensor, Tensor> _fc1;
        private readonly Module<Tensor, Tensor> _fc2;

        // Слой Dropout для регуляризации.
        // "Выключает" случайные нейроны во время обучения, чтобы предотвратить переобучение.
        private readonly Module<Tensor, Tensor> _dropout;

        // 3. Слои для механизма внимания (Attention Mechanism)
        // Эти слои позволяют модели фокусироваться на наиболее важных словах в предложении.
        private readonly Module<Tensor, Tensor> _attentionLinear;
        private readonly Parameter _attentionContextVector;

        public IntentClassifier(int vocabSize, int embDim, int hiddenSize, int numLayers, int numLabels) : base(nameof(IntentClassifier))
        {
            // Инициализация всех слоев, которые будут использоваться в модели.

            _embedding = Embedding(vocabSize, embDim);
            _lstm = LSTM(embDim, hiddenSize, numLayers: numLayers, batchFirst: true, bidirectional: true, dropout: 0.2);

            // Выход LSTM удваивается, так как он двунаправленный (bidirectional).
            int lstm_output_dim = hiddenSize * 2;

            // Инициализация слоев для механизма внимания.
            _attentionLinear = Linear(lstm_output_dim, lstm_output_dim);
            _attentionContextVector = Parameter(randn(lstm_output_dim)); // Обучаемый вектор для вычисления "важности".

            // Инициализация полносвязных слоев.
            _fc1 = Linear(lstm_output_dim, 128);
            _fc2 = Linear(128, numLabels);
            _dropout = Dropout(0.5);

            // Регистрация всех компонентов модели, чтобы PyTorch мог отслеживать их параметры.
            RegisterComponents();
        }

        /// <summary>
        /// Метод Forward определяет, как данные проходят через слои нейросети.
        /// </summary>
        /// <param name="data">Кортеж, содержащий входной тензор с ID токенов и тензор маски.</param>
        /// <returns>Тензор с логитами для каждого класса.</returns>
        public override Tensor forward((Tensor input, Tensor mask) data)
        {
            var (inputIds, mask) = data;

            // Шаг 1: Прогоняем ID токенов через слой эмбеддингов.
            // Размерность: [batch, seq_len] -> [batch, seq_len, embDim]
            Tensor x = _embedding.forward(inputIds);

            // Шаг 2: Прогоняем эмбеддинги через LSTM.
            // Размерность: [batch, seq_len, embDim] -> [batch, seq_len, 2 * hiddenSize]
            var (lstmOutput, _, _) = _lstm.forward(x);

            // --- Шаг 3: Механизм Внимания (Attention) ---
            // Этот блок вычисляет, на какие слова в предложении нужно обратить больше внимания.

            // 3.1. Применяем линейное преобразование и функцию активации tanh.
            // Это помогает извлечь признаки для вычисления "важности".
            using Tensor u = functional.tanh(_attentionLinear.forward(lstmOutput));

            // 3.2. Вычисляем "оценки важности" (scores) для каждого слова.
            // Мы умножаем преобразованный выход LSTM на обучаемый контекстный вектор.
            using Tensor scores = matmul(u, _attentionContextVector);

            // 3.3. Маскируем padding-токены перед Softmax.
            // Мы не хотим, чтобы модель обращала внимание на "пустые" токены.
            // Заменяем оценки для padding-токенов на очень маленькое число.
            using Tensor masked_scores = scores.masked_fill(mask.eq(0), float.NegativeInfinity);

            // 3.4. Применяем Softmax, чтобы превратить оценки в "веса внимания" (от 0 до 1).
            // Сумма всех весов для одного предложения будет равна 1.
            using Tensor alpha = functional.softmax(masked_scores, dim: 1);

            // 3.5. Вычисляем контекстный вектор.
            // Это взвешенная сумма выходов LSTM, где весами являются полученные "веса внимания".
            // В результате мы получаем один вектор, который представляет все предложение,
            // но с акцентом на самых важных словах.
            using Tensor context_vector = sum(alpha.unsqueeze(-1) * lstmOutput, dim: 1);
            // --- Конец механизма внимания ---

            // Шаг 4: Прогоняем контекстный вектор через полносвязные слои.
            Tensor y = functional.relu(_fc1.forward(context_vector));
            y = _dropout.forward(y); // Применяем Dropout для регуляризации
            y = _fc2.forward(y);     // Финальный слой, выдающий логиты.

            return y;
        }
    }
}