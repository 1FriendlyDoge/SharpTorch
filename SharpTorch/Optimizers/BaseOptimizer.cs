namespace SharpTorch.Optimizers;

public abstract class BaseOptimizer(int maxDatapoints)
{
    private readonly List<float> MostRecentLosses = [];
    protected abstract float OptimizeImplementation(List<float> mostRecentLosses, float learningRateCap);
    
    public float Optimize(float loss, float learningRateCap)
    {
        MostRecentLosses.Add(loss);
        if (MostRecentLosses.Count > maxDatapoints)
        {
            MostRecentLosses.RemoveAt(0);
        }
        
        return OptimizeImplementation(MostRecentLosses, learningRateCap);
    }
}