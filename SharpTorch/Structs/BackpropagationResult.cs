namespace SharpTorch.Structs;

public class BackpropagationResult
{
    public float[,] WeightGradients { get; init; }
    public float[] BiasGradients { get; init; }
    
    public BackpropagationResult(float[,] weightGradients, float[] biasGradients)
    {
        WeightGradients = weightGradients;
        BiasGradients = biasGradients;
    }
}