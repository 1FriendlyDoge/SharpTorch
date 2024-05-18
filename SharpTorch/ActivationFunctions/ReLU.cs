namespace SharpTorch.ActivationFunctions;

public class ReLU : BaseActivation
{
    public override float Activate(float input)
    {
        return Math.Max(0, input);
    }

    public override float CalculateDerivative(float input)
    {
        return input > 0 ? 1 : 0;
    }
}