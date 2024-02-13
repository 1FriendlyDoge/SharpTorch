namespace SharpTorch.Exceptions;

public class TensorDimensionsMismatchException : Exception
{
    public TensorDimensionsMismatchException() { }
    
    public TensorDimensionsMismatchException(string message) : base(message) { }
    
    public TensorDimensionsMismatchException(string message, Exception innerException) : base(message, innerException) { }
}