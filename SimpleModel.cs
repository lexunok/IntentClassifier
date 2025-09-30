using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

public class SimpleClassifier : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> embedding;
    private readonly Module<Tensor, Tensor> fc;

    public SimpleClassifier(int vocabSize, int embDim, int numLabels) : base(nameof(SimpleClassifier))
    {
        embedding = Embedding(vocabSize, embDim);
        fc = Linear(embDim, numLabels);
        RegisterComponents(); // ОБЯЗАТЕЛЬНО
    }

    public override Tensor forward(Tensor input)
    {
        var x = embedding.forward(input).mean([1]);
        return fc.forward(x);
    }
}