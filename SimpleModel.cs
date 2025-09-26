using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

public class SimpleClassifier : Module<Tensor,Tensor>
{
    private Module<Tensor,Tensor> embedding;
    private Module<Tensor,Tensor> linear;

    public SimpleClassifier(string name, int vocabSize, int embDim, int numLabels) : base(name)
    {
        embedding = Embedding(vocabSize, embDim);
        linear = Linear(embDim, numLabels);
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        // input: (batch, seq_len) int64
        var x = embedding.forward(input).to_type(ScalarType.Float32); // (B, L, E)
        var mean = x.mean([1]); // (B, E)
        var logits = linear.forward(mean);
        return logits;
    }
}