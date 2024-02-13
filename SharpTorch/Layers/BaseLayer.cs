namespace SharpTorch.Layers;

public abstract class BaseLayer
{
    protected int InputSize { get; init; }
    protected int OutputSize { get; init; }

    protected BaseLayer(int inputSize, int outputSize)
    {
        InputSize = inputSize;
        OutputSize = outputSize;
    }

    public abstract float[] Forward(float[] input);
}