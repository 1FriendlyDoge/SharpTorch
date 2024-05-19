using SharpTorch.ActivationFunctions;
using SharpTorch.Layers;
using SharpTorch.Models;

namespace SharpTorchSamples;

public class TestModel : BaseModel
{
    public TestModel()
    {
        Layers =
        [
            new LinearLayer(1, 64, new ReLU()),
            new LinearLayer(64, 128, new ReLU()),
            new LinearLayer(128, 1, new ReLU())
        ];
    }
}