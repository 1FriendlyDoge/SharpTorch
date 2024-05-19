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
            new LinearLayer(1, 10, new ReLU()),
            new LinearLayer(10, 100, new ReLU()),
            new LinearLayer(100, 1, new ReLU())
        ];
    }
}