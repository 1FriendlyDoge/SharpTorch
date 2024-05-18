namespace SharpTorch.Losses;

public class MeanSquaredError : BaseLoss
{
    protected override float Calculate(float yPredicted, float yResult)
    {
        float diff = yPredicted - yResult;
        return diff * diff;
    }

    protected override float CalculateDerivative(float yPredicted, float yResult)
    {
        return 2.0f * (yPredicted - yResult) / 1;
    }
}