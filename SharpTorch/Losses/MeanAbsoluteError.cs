namespace SharpTorch.Losses;

public class MeanAbsoluteError : BaseLoss
{
    protected override float Calculate(float yPredicted, float yResult)
    {
        return Math.Abs(yPredicted - yResult);
    }

    protected override float CalculateDerivative(float yPredicted, float yResult)
    {
        if (yPredicted < yResult)
        {
            return -1;
        }

        return yPredicted > yResult ? 1 : 0;
    }
}