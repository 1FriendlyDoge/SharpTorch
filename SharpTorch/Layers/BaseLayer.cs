using SharpTorch.Models;

namespace SharpTorch.Layers;

public abstract class BaseLayer
{
    protected internal bool TrainMode { get; set; } = true;
    
    public int InputSize { get; init; }
    public int OutputSize { get; init; }
    
    public float[,] Weights = new float[0, 0];
    public float[] Biases = [];
    
    public float[] Inputs { get; private set; } = [];

    protected BaseLayer(int inputSize, int outputSize)
    {
        InputSize = inputSize;
        OutputSize = outputSize;
    }

    public float[] Forward(float[] input)
    {
        if (TrainMode)
        {
            Inputs = input;
        }
        
        return ForwardImplementation(input);
    }

    protected abstract float[] ForwardImplementation(float[] input);
}