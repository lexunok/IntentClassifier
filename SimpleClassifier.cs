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

    // Attention layers
    private readonly Module<Tensor, Tensor> attention_linear;
    private readonly Parameter attention_context_vector;

    public SimpleClassifier(int vocabSize, int embDim, int hiddenSize, int numLayers, int numLabels) : base(nameof(SimpleClassifier))
    {
        embedding = Embedding(vocabSize, embDim);
        lstm = LSTM(embDim, hiddenSize, numLayers: numLayers, batchFirst: true, bidirectional: true, dropout: 0.2);
        
        // LSTM is bidirectional, so output is 2 * hiddenSize
        int lstm_output_dim = hiddenSize * 2;

        // Attention layers
        attention_linear = Linear(lstm_output_dim, lstm_output_dim);
        attention_context_vector = Parameter(torch.randn(lstm_output_dim));

        fc1 = Linear(lstm_output_dim, 128); 
        fc2 = Linear(128, numLabels);
        dropout = Dropout(0.5);
        RegisterComponents();
    }

    public override Tensor forward((Tensor input, Tensor mask) data)
    {
        var (inputIds, mask) = data;

        // embedding: [batch, seq_len] -> [batch, seq_len, embDim]
        var x = embedding.forward(inputIds);

        // lstm: [batch, seq_len, embDim] -> (lstmOutput: [batch, seq_len, 2 * hiddenSize], ...)
        var (lstmOutput, _, _) = lstm.forward(x);

        // --- Attention Mechanism ---
        // Apply linear transformation to LSTM output: [batch, seq_len, lstm_output_dim]
        using var u = functional.tanh(attention_linear.forward(lstmOutput));

        // Calculate attention scores: [batch, seq_len]
        // attention_context_vector: [lstm_output_dim]
        // u: [batch, seq_len, lstm_output_dim]
        // scores: [batch, seq_len]
        using var scores = torch.matmul(u, attention_context_vector);

        // Mask padding tokens before softmax
        // mask: [batch, seq_len]
        // scores: [batch, seq_len]
        // Where mask is 0 (padding), set scores to a very small negative number
        using var masked_scores = scores.masked_fill(mask.eq(0), float.NegativeInfinity);

        // Apply softmax to get attention weights: [batch, seq_len]
        using var alpha = functional.softmax(masked_scores, dim: 1);

        // Apply attention weights to LSTM output to get context vector: [batch, lstm_output_dim]
        // alpha: [batch, seq_len, 1] (unsqueeze for broadcasting)
        // lstmOutput: [batch, seq_len, lstm_output_dim]
        // context_vector: [batch, lstm_output_dim]
        using var context_vector = torch.sum(alpha.unsqueeze(-1) * lstmOutput, dim: 1);
        // --- End Attention Mechanism ---

        // Полносвязные слои
        var y = functional.relu(fc1.forward(context_vector));
        y = dropout.forward(y);
        y = fc2.forward(y);
        return y;
    }
}
