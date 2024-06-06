using SharpTorch.ActivationFunctions;
using SharpTorch.Layers;

namespace SharpTorch.Models;

public class LinearSequentialNetwork : BaseModel
{
    public LinearSequentialNetwork(params (int, int, BaseActivation?)[] layers)
    {
        List<BaseLayer> layersList = [];
        layersList.AddRange(layers.Select(layer
            => new LinearLayer(layer.Item1, layer.Item2, layer.Item3)));
        
        Layers = layersList.ToArray();
    }
}