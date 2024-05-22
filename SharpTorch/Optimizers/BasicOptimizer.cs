namespace SharpTorch.Optimizers;

public class BasicOptimizer(int maxDatapointCount = 50) : BaseOptimizer(maxDatapointCount)
{
    private readonly List<float> LossDeltas = [];
    private readonly int _maxDatapointCount = maxDatapointCount;

    
    // TODO: maybe find a better method later, this somehow makes the loss worse (???)
    protected override float OptimizeImplementation(List<float> mostRecentLosses, float learningRateCap)
    {
        if (mostRecentLosses.Count < 2)
        {
            return learningRateCap;
        }
        
        float lossDelta = mostRecentLosses[^1] - mostRecentLosses[^2];
        LossDeltas.Add(lossDelta);
        if (LossDeltas.Count > _maxDatapointCount)
        {
            LossDeltas.RemoveAt(0);
        }
        float sum = LossDeltas.Sum();
        float averageAscentRate = sum / LossDeltas.Count;

        if (averageAscentRate < 0.01F)
        {
            return learningRateCap;
        }

        float lr = 0.5F * learningRateCap;
        Console.WriteLine($"Learning rate: {lr}");
        return lr > learningRateCap ? learningRateCap : lr;
    }
}