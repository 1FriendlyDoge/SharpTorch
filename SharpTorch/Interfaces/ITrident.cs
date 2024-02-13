namespace SharpTorch.Interfaces;

public interface ITrident<out T>
{
    public T Clone();
}