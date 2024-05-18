namespace SharpTorch;

public static class Utils
{
    public static float[] Project1D(float[,] input, int index)
    {
        float[] output = new float[input.GetLength(1)];
        for (int i = 0; i < input.GetLength(1); i++)
        {
            output[i] = input[index, i];
        }

        return output;
    }
    
    public static float[,] Project2D(float[,,] input, int index)
    {
        float[,] output = new float[input.GetLength(1), input.GetLength(2)];
        for (int i = 0; i < input.GetLength(1); i++)
        {
            for (int j = 0; j < input.GetLength(2); j++)
            {
                output[i, j] = input[index, i, j];
            }
        }

        return output;
    }
    
    public static float[,] ConvertTo2D(List<float[]> input)
    {
        float[,] output = new float[input.Count, input[0].Length];
        for (int i = 0; i < input.Count; i++)
        {
            for (int j = 0; j < input[0].Length; j++)
            {
                output[i, j] = input[i][j];
            }
        }

        return output;
    }
}