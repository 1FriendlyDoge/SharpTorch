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
            new LinearLayer(1, 700, new ReLU()),
            new LinearLayer(700, 1, new ReLU())
        ];
    }
}