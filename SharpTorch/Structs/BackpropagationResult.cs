using SharpTorch.Layers;
using SharpTorch.Models;

namespace SharpTorch.Structs;

public class BackpropagationResult
{
    public List<float[,]> WeightGradients { get; }
    public List<float[]> BiasGradients { get; }
    
    public BackpropagationResult(List<float[,]> weightGradients, List<float[]> biasGradients)
    {
        WeightGradients = weightGradients;
        BiasGradients = biasGradients;
    }

    public BackpropagationResult(BaseModel reference)
    {
        WeightGradients = new List<float[,]>();
        BiasGradients = new List<float[]>();
        
        foreach (BaseLayer layer in reference.Layers)
        {
            WeightGradients.Add(new float[layer.Weights.GetLength(0), layer.Weights.GetLength(1)]);
            BiasGradients.Add(new float[layer.Biases.Length]);
        }
    }
    
    public void Add(BackpropagationResult result)
    {
        for (int i = 0; i < WeightGradients.Count; i++)
        {
            for (int x = 0; x < WeightGradients[i].GetLength(0); x++)
            {
                for (int y = 0; y < WeightGradients[i].GetLength(1); y++)
                {
                    WeightGradients[i][x, y] += result.WeightGradients[i][x, y];
                }
            }
            
            for (int x = 0; x < BiasGradients[i].Length; x++)
            {
                BiasGradients[i][x] += result.BiasGradients[i][x];
            }
        }
    }
    
    public void Average(int batchSize)
    {
        for (int i = 0; i < WeightGradients.Count; i++)
        {
            for (int x = 0; x < WeightGradients[i].GetLength(0); x++)
            {
                for (int y = 0; y < WeightGradients[i].GetLength(1); y++)
                {
                    WeightGradients[i][x, y] /= batchSize;
                }
            }
            
            for (int x = 0; x < BiasGradients[i].Length; x++)
            {
                BiasGradients[i][x] /= batchSize;
            }
        }
    }
}