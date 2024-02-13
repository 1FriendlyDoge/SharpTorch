namespace SharpTorch.Losses;

public class SquaredAbsoluteLoss : BaseLoss
{
    public override float Calculate(float[] x, float[] y)
    {
        float loss = 0;
        
        for (int i = 0; i < x.Length; i++)
        {
            loss += MathF.Pow(MathF.Max(x[i], y[i]) - MathF.Min(x[i], y[i]), 2);
        }
        
        return loss;
    }
}