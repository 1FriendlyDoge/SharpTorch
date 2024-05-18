namespace SharpTorch.Interfaces;

public interface IClone<out T>
{
    public T Clone();
}