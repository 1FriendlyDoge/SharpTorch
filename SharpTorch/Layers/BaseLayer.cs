﻿using SharpTorch.ActivationFunctions;

namespace SharpTorch.Layers;

public abstract class BaseLayer
{
    protected internal bool TrainMode { get; set; } = true;
    
    public int InputSize { get; set; }
    public int OutputSize { get; set; }

    public BaseActivation? ActivationFunction { get; set; }
    
    public float[,] Weights = new float[0, 0];
    public float[] Biases = [];
    
    public float[] Inputs { get; private set; } = [];
    public float[] RawOutputs { get; private set; } = [];

    protected BaseLayer(int inputSize, int outputSize, BaseActivation? activationFunction = null)
    {
        InputSize = inputSize;
        OutputSize = outputSize;
        ActivationFunction = activationFunction;
    }

    public float[] Forward(float[] input)
    {
        float[] values = ForwardImplementation(input);
        
        if (TrainMode)
        {
            Inputs = input;
            RawOutputs = values;
        }
        
        if (ActivationFunction == null)
        {
            return values;
        }

        for (int i = 0; i < values.Length; i++)
        {
            values[i] = ActivationFunction.Activate(values[i]);
        }

        return values;
    }

    protected abstract float[] ForwardImplementation(float[] input);
}