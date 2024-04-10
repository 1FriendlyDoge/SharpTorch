using System.Runtime.Serialization.Formatters.Binary;

namespace SharpTorch.Statics;

public class Func
{
    public static T DeepCopy<T>(T obj) where T : class
    {
        using (MemoryStream stream = new())
        {
            BinaryFormatter formatter = new();
            formatter.Serialize(stream, obj);
            stream.Seek(0, SeekOrigin.Begin);
            return (T)formatter.Deserialize(stream);
        }
    }
}