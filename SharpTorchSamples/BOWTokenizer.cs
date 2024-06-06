namespace SharpTorchSamples;

public class BOWTokenizer
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
    
    public int[] Tokenize(string input)
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
        
        int[] output = new int[TokenCollection.Count];
        for (int i = 0; i < output.Length; i++)
        {
            output[i] = 0;
        }

        foreach (int token in tokens)
        {
            output[token - 1] += 1;
        }
        
        return output;
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
        return sequence.Split(' ', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries);
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