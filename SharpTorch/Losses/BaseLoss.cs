namespace SharpTorch.Losses;

public abstract class BaseLoss
{
    public abstract float Calculate(float[] x, float[] y);
}