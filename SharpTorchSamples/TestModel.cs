using SharpTorch.Layers;
using SharpTorch.Models;

namespace SharpTorchSamples;

public class TestModel : BaseModel
{
    public TestModel()
    {
        Layers =
        [
            new LinearLayer(1, 1)
        ];
    }
}