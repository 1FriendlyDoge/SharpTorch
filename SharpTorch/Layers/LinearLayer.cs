﻿using SharpTorch.ActivationFunctions;

namespace SharpTorch.Layers;

public class LinearLayer : BaseLayer
{
    public LinearLayer(int inputSize, int outputSize, BaseActivation? activation = null, bool randomizedWeights = true) : base(inputSize, outputSize, activation)
    {
        Weights = new float[inputSize, outputSize];
        Biases = new float[outputSize];
        
        if (randomizedWeights)
        {
            RandomizeWeights();
        }
        else
        {
            for (int i = 0; i < Weights.GetLength(0); i++)
            {
                for (int x = 0; x < Weights.GetLength(1); x++)
                {
                    Weights[i, x] = 0.001F;
                }
            }
            for (int i = 0; i < Biases.Length; i++)
            {
                Biases[i] = 0.001F;
            }
        }
    }

    protected override float[] ForwardImplementation(float[] input)
    {
        float[] output = new float[OutputSize];

        for (int i = 0; i < OutputSize; i++)
        {
            float sum = 0;
            for (int j = 0; j < InputSize; j++)
            {
                sum += input[j] * Weights[j, i];
            }

            output[i] = sum + Biases[i];
        }
        
        return output;
    }

    private void RandomizeWeights(int seed = 0)
    {
        Random random = new(seed);
        for (int i = 0; i < Weights.GetLength(0); i++)
        {
            for (int x = 0; x < Weights.GetLength(1); x++)
            {
                Weights[i, x] = random.NextSingle();
            }
        }
        for (int i = 0; i < Biases.Length; i++)
        {
            Biases[i] = random.NextSingle();
        }
    }
}