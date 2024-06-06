namespace SharpTorchSamples;

public class SequenceTokenizer
{
    private readonly Dictionary<string, int> TokenCollection = new();
    private bool FitMode = true;

    public void Lock()
    {
        FitMode = false;
    }

    public void Unlock()
    {
        FitMode = true;
    }
    
    public int[] Tokenize(string input, int? paddingLength = null)
    {
        string[] sequences = Seperate(input);
        
        List<int> tokens = [];
        // ReSharper disable once LoopCanBeConvertedToQuery
        foreach (string token in sequences)
        {
            int? encoded = GetOrAdd(token);
            if (encoded != null)
            {
                tokens.Add(encoded.Value);
            }
        }

        if (paddingLength == null)
        {
            return tokens.ToArray();
        }

        tokens = tokens.Take(paddingLength.Value).ToList();
        
        while (tokens.Count < paddingLength.Value)
        {
            tokens.Add(-1);
        }

        return tokens.ToArray();
    }

    private int? GetOrAdd(string sequence)
    {
        if (TokenCollection.TryGetValue(sequence.ToUpper(), out int value))
        {
            return value;
        }

        if (!FitMode)
        {
            return null;
        }

        int idx = TokenCollection.Count + 1;
        TokenCollection.Add(sequence.ToUpper(), idx);
        return idx;
    }
    
    private string[] Seperate(string sequence)
    {
        return sequence.Split(' ');
    }
    
    public static float[] SequenceEncode(int[] sequence)
    {
        float[] output = new float[sequence.Length];
        for (int i = 0; i < sequence.Length; i++)
        {
            output[i] = sequence[i];
        }

        return output;
    }
}