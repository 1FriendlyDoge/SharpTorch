namespace SharpTorch.Losses;

public class MultiLabelCrossEntropyLoss : BaseLoss
{
    protected override float Calculate(float yPredicted, float yResult)
    {
        return -yResult * (float)Math.Log(yPredicted) - (1 - yResult) * (float)Math.Log(1 - yPredicted);
    }

    protected override float CalculateDerivative(float yPredicted, float yResult)
    {
        return -(yResult / yPredicted) + (1 - yResult) / (1 - yPredicted);
    }
}