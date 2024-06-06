using CsvHelper.Configuration.Attributes;

namespace SharpTorchSamples;

public class DataPoint
{
    [Name("text")]
    public string Text { get; set; }
    
    [Name("spam")]
    public int Spam { get; set; }
}