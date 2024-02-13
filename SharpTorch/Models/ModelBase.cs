using SharpTorch.Layers;

namespace SharpTorch.Models;

public abstract class ModelBase<T> : Interfaces.ITrident<T>
{
    public BaseLayer[] Layers { get; set; }

    protected ModelBase(BaseLayer[]? layers = null)
    {
        Layers = layers ?? [];
    }

    public float[] Forward(float[] input)
    {
        foreach (BaseLayer layer in Layers)
        {
            input = layer.Forward(input);
        }
        
        return input;
    }

    public abstract T Clone();
}