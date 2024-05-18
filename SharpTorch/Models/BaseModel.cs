using SharpTorch.Layers;
using SharpTorch.Structs;

namespace SharpTorch.Models;

public abstract class BaseModel
{
    public BaseLayer[] Layers { get; set; }
    public bool TrainMode { get; private set; } = true;

    protected BaseModel(BaseLayer[]? layers = null)
    {
        Layers = layers ?? [];

        foreach (BaseLayer layer in Layers)
        {
            layer.TrainMode = TrainMode;
        }
    }

    public void Eval()
    {
        TrainMode = false;
        SetLayerMode();
    }

    public void Train()
    {
        TrainMode = true;
        SetLayerMode();
    }

    private void SetLayerMode()
    {
        foreach (BaseLayer layer in Layers)
        {
            layer.TrainMode = TrainMode;
        }
    }

    public float[] Forward(float[] input)
    {
        foreach (BaseLayer layer in Layers)
        {
            input = layer.Forward(input);
        }
        
        return input;
    }

    public void UpdateValues(BackpropagationResult values, float learningRate)
    {
        for (int i = 0; i < Layers.Length; i++)
        {
            BaseLayer layer = Layers[i];
            float[,] weightGradients = values.WeightGradients[i];
            float[] biasGradients = values.BiasGradients[i];
            
            for (int x = 0; x < layer.Weights.GetLength(0); x++)
            {
                for (int y = 0; y < layer.Weights.GetLength(1); y++)
                {
                    layer.Weights[x, y] -= weightGradients[x, y] * learningRate;
                }
            }
            
            for (int x = 0; x < layer.Biases.Length; x++)
            {
                layer.Biases[x] -= biasGradients[x] * learningRate;
            }
        }
    }
}