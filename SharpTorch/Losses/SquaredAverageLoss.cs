namespace SharpTorch.Losses;

public class SquaredAverageLoss : BaseLoss
{
    public override float Calculate(float[] x, float[] y)
    {
        if(x.Length == 0)
        {
            return 0;
        }
        
        float loss = 0;
        
        for (int i = 0; i < x.Length; i++)
        {
            loss += MathF.Pow(MathF.Max(x[i], y[i]) - MathF.Min(x[i], y[i]), 2);
        }
        
        return loss / x.Length;
    }
}