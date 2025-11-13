using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

public class SimpleClassifier : Module<(Tensor input, Tensor mask), Tensor>
{
    private readonly Module<Tensor, Tensor> embedding;
    private readonly LSTM lstm;
    private readonly Module<Tensor, Tensor> fc1;
    private readonly Module<Tensor, Tensor> fc2;
    private readonly Module<Tensor, Tensor> dropout;

    public SimpleClassifier(int vocabSize, int embDim, int hiddenSize, int numLabels) : base(nameof(SimpleClassifier))
    {
        embedding = Embedding(vocabSize, embDim);
        lstm = LSTM(embDim, hiddenSize, batchFirst: true, bidirectional: true);
        // LSTM is bidirectional, so output is 2 * hiddenSize
        fc1 = Linear(hiddenSize * 2, 128);
        fc2 = Linear(128, numLabels);
        dropout = Dropout(0.5);
        RegisterComponents();
    }

    public override Tensor forward((Tensor input, Tensor mask) data)
    {
        var (inputIds, mask) = data;

        // embedding: [batch, seq_len] -> [batch, seq_len, embDim]
        var x = embedding.forward(inputIds);

        // lstm: [batch, seq_len, embDim] -> [batch, seq_len, 2 * hiddenSize]
        var (lstmOutput, _, _) = lstm.forward(x);

        // We want the output of the last token for each sequence.
        // We can use the mask to find the length of each sequence.
        var lengths = mask.sum(1).to(ScalarType.Int64) - 1; // [batch]

        // Index into the lstmOutput to get the last relevant output
        // lstmOutput is [batch, seq_len, hidden_size * 2]
        // We need to gather the elements at the specified lengths.
        // The shape of lengths is [batch], we need to make it [batch, 1, hidden_size * 2]
        // to use it with gather.
        var lastTokenIndices = lengths.view(lengths.shape[0], 1, 1).expand(-1, -1, lstmOutput.shape[2]);
        var lastTokenOutputs = lstmOutput.gather(1, lastTokenIndices).squeeze(1);

        // Полносвязные слои
        var y = functional.relu(fc1.forward(lastTokenOutputs));
        y = dropout.forward(y);
        y = fc2.forward(y);
        return y;
    }
}
