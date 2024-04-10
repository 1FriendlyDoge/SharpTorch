using SharpTorch.Layers;
using SharpTorch.Statics;

namespace SharpTorch.Models;

public class TestModel : ModelBase<TestModel>
{
    public override TestModel Clone()
    {
        return Func.DeepCopy(this);
    }
}