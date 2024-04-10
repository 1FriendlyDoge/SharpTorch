namespace SharpTorch.Layers;

public class LinearLayer : BaseLayer
{
    private readonly float[] weights;
    private readonly float[] biases;
    
    public LinearLayer(int inputSize, int outputSize, bool randomizedWeights = true) : base(inputSize, outputSize)
    {
        weights = new float[inputSize * outputSize];
        biases = new float[outputSize];
        
        if (randomizedWeights)
        {
            RandomizeWeights();
        }
    }

    public override float[] Forward(float[] input)
    {
        float[] output = new float[OutputSize];
        
        for (int i = 0; i < OutputSize; i++)
        {
            float sum = 0;
            for (int j = 0; j < InputSize; j++)
            {
                sum += input[j] * weights[j + i * InputSize];
            }
            output[i] = sum + biases[i];
        }
        
        return output;
    }

    private void RandomizeWeights()
    {
        Random random = new();
        for (int i = 0; i < weights.Length; i++)
        {
            weights[i] = random.NextSingle();
        }
        for (int i = 0; i < biases.Length; i++)
        {
            biases[i] = random.NextSingle();
        }
    }
}