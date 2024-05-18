namespace SharpTorch.ActivationFunctions;

public abstract class BaseActivation
{
    public abstract float Activate(float input);
    public abstract float CalculateDerivative(float input);
}