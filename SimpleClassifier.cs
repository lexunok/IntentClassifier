using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;


public class SimpleClassifier : Module<(Tensor input, Tensor mask), Tensor>
{
    private readonly Module<Tensor, Tensor> embedding;
    private readonly Module<Tensor, Tensor> fc1;
    private readonly Module<Tensor, Tensor> fc2;
    private readonly Module<Tensor, Tensor> dropout;

    public SimpleClassifier(int vocabSize, int embDim, int numLabels) : base(nameof(SimpleClassifier))
    {
        embedding = Embedding(vocabSize, embDim);
        fc1 = Linear(embDim, 128);
        fc2 = Linear(128, numLabels);
        dropout = Dropout(0.3);
        RegisterComponents();
    }

    // forward принимает tuple (input, mask)
    public override Tensor forward((Tensor input, Tensor mask) data)
    {
        var (inputIds, mask) = data;

        // embedding
        var x = embedding.forward(inputIds); // [batch, seq_len, embDim]

        // Маскируем паддинги
        var maskFloat = mask.to_type(ScalarType.Float32).unsqueeze(-1); // [batch, seq_len, 1]
        x = x * maskFloat;

        // Суммируем по seq_len и делим на количество не паддингов
        var lengths = maskFloat.sum(1).clamp_min(1.0f); // [batch, 1]
        x = x.sum(1) / lengths;

        // Полносвязные слои
        x = functional.relu(fc1.forward(x));
        x = dropout.forward(x);
        x = fc2.forward(x);
        return x;
    }
}