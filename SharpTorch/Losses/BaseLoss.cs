namespace SharpTorch.Losses;

public abstract class BaseLoss
{
    protected abstract float Calculate(float yPredicted, float yResult);
    protected abstract float CalculateDerivative(float yPredicted, float yResult);

    public float CalculateAll(float[] yPredicted, float[] yResult)
    {
        float sum = 0;
        for (int i = 0; i < yResult.Length; i++)
        {
            sum += Calculate(yPredicted[i], yResult[i]);
        }
        
        return sum / yResult.Length;
    }
    
    public float[] GradientAll(float[] yPredicted, float[] yResult)
    {
        float[] gradients = new float[yResult.Length];
        for (int i = 0; i < yResult.Length; i++)
        {
            gradients[i] = CalculateDerivative(yPredicted[i], yResult[i]);

            if (gradients[i] > 50000)
            {
                throw new Exception();
            }
        }
        
        return gradients;
    }
}