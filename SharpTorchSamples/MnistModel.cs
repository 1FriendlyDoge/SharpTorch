using SharpTorch.ActivationFunctions;
using SharpTorch.Layers;
using SharpTorch.Models;

namespace SharpTorchSamples;

public class MnistModel : BaseModel
{
    public MnistModel()
    {
        Layers = new BaseLayer[]
        {
            new LinearLayer(784, 128, new ReLU()),
            new LinearLayer(128, 10)
        };
    }
}