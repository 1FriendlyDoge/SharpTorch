namespace SharpTorch.Structs;

public class BackpropagationResult
{
    public List<float[,]> WeightGradients { get; init; }
    public List<float[]> BiasGradients { get; init; }
    
    public BackpropagationResult(List<float[,]> weightGradients, List<float[]> biasGradients)
    {
        WeightGradients = weightGradients;
        BiasGradients = biasGradients;
    }
}