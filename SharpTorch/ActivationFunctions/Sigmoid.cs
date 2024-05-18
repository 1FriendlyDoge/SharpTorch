namespace SharpTorch.ActivationFunctions;

public class Sigmoid : BaseActivation
{
    public override float Activate(float input)
    {
        return 1 / (1 + MathF.Exp(-input));
    }

    public override float CalculateDerivative(float input)
    {
        float sigmoid = Activate(input);
        return sigmoid * (1 - sigmoid);
    }
}