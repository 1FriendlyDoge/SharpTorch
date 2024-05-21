using SharpTorch.Layers;
using SharpTorch.Structs;

namespace SharpTorch.Models;

public abstract class BaseModel
{
    public BaseLayer[] Layers { get; protected set; }
    private bool TrainMode { get; set; } = true;

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

    public void DebugPrint()
    {
        for (int i = 0; i < Layers.Length; i++)
        {
            BaseLayer layer = Layers[i];
            Console.WriteLine($"Layer {i + 1}");
            Console.WriteLine("\tWeights:");
            Console.Write("\t\t");
            for (int x = 0; x < layer.Weights.GetLength(0); x++)
            {
                for (int y = 0; y < layer.Weights.GetLength(1); y++)
                {
                    Console.Write($"{layer.Weights[x, y]:0.0000} ");
                }
            }
            Console.WriteLine();

            Console.WriteLine("\tBiases:");
            Console.Write("\t\t");
            foreach (var t in layer.Biases)
            {
                Console.Write($"{t:0.0000} ");
            }
            Console.WriteLine();
        }
    }
}