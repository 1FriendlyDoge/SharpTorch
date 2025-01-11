using SharpTorch.ActivationFunctions;
using SharpTorch.Layers;
using SharpTorch.Losses;
using SharpTorch.Models;
using SharpTorch.Optimizers;
using SharpTorch.Structs;

namespace SharpTorch;

public class Trainer
{
    private BaseModel Model { get; set; }
    private BaseLoss Loss { get; set; }
    private float LearningRate { get; set; }
    private float InitalLearningRate { get; set; }
    private BaseOptimizer? Optimizer { get; set; }
    private int BatchSize { get; set; }
    private int Epochs { get; set; }
    private float[,] X { get; set; }
    private float[,] Y { get; set; }
    private int ValidationInterval { get; set; }
    private int DisplayValidationInterval { get; set; }
    private float GradientClip { get; set; }
    private CancellationTokenSource? CancellationTokenSource { get; set; }
    
    public Trainer(BaseModel model, BaseLoss loss, float[,] x, float[,] y, float learningRate = 0.01f, BaseOptimizer? optimizer = null, int batchSize = 1, int epochs = 10, int validationInterval = 1, int displayValidationInterval = 5, float gradientClip = 1.0F)
    {
        Model = model;
        Loss = loss;
        LearningRate = learningRate;
        InitalLearningRate = learningRate;
        Optimizer = optimizer;
        BatchSize = batchSize;
        Epochs = epochs;
        X = x;
        Y = y;
        ValidationInterval = validationInterval;
        DisplayValidationInterval = displayValidationInterval;
        GradientClip = gradientClip;
    }
    
    public void Train(CancellationTokenSource? cts = null)
    {
        cts ??= new CancellationTokenSource();
        CancellationTokenSource = cts;
        Console.WriteLine("Training...");
        for (int epoch = 0; epoch < Epochs; epoch++)
        {
            for (int dataIndex = 0; dataIndex < X.GetLength(0); dataIndex += BatchSize)
            {
                if (cts is {IsCancellationRequested: true})
                {
                    return;
                }
                
                Task?[] tasks = new Task[BatchSize];
                BackpropagationResult backpropagationResult = new(Model);
                for (int batchIndex = 0; batchIndex < BatchSize; batchIndex++)
                {
                    if (dataIndex + batchIndex >= X.GetLength(0))
                    {
                        break;
                    }
                    
                    int index = dataIndex + batchIndex;

                    Model.Train();
                    tasks[batchIndex] = Task.Run(() =>
                    {
                        float[] xData = Utils.Project1D(X, index);
                        float[] yResult = Utils.Project1D(Y, index);
                        
                        float[] yPredicted = Model.Forward(xData);
                        
                        BackpropagationResult result = Backward(yPredicted, yResult, Loss);
                        
                        backpropagationResult.Add(result);
                    }, cts.Token);
                }
                
                foreach (Task? t in tasks)
                {
                    t?.Wait();
                }
                
                backpropagationResult.Average(BatchSize);
                Model.UpdateValues(backpropagationResult, LearningRate);
            }
            
            if (epoch % ValidationInterval == 0)
            {
                RunValidation(epoch);
            }
        }
    }

    private void RunValidation(int epoch)
    {
        Model.Eval();
        float totalLoss = 0;
        Parallel.For(0, X.GetLength(0), i =>
        {
            float[] xData = Utils.Project1D(X, i);
            float[] yResult = Utils.Project1D(Y, i);
            
            float[] yPredicted = Model.Forward(xData);
            totalLoss += Loss.CalculateAll(yPredicted, yResult);
        });
        float actualLoss = totalLoss / X.GetLength(0);
        
        if (Optimizer != null)
        {
            LearningRate = Optimizer.Optimize(totalLoss, InitalLearningRate);
        }

        if (epoch % DisplayValidationInterval == 0)
        {
            Console.WriteLine($"Epoch: {epoch}/{Epochs}, Loss: {actualLoss}, LR: {LearningRate}");
        }

        if (actualLoss != 0)
        {
            return;
        }

        CancellationTokenSource?.Cancel();
        Console.WriteLine("Loss is 0, stopping training...");
    }

    private BackpropagationResult Backward(float[] yPredicted, float[] yResult, BaseLoss loss)
    {
        List<float[,]> weightGradients = [];
        List<float[]> biasGradients = [];
        
        foreach (BaseLayer t in Model.Layers)
        {
            weightGradients.Add(new float[t.InputSize, t.OutputSize]);
            biasGradients.Add(new float[t.OutputSize]);
        }
        
        float[] outputGradients = loss.GradientAll(yPredicted, yResult);
    
        for (int i = Model.Layers.Length - 1; i >= 0; i--)
        {
            float[] inputGradients = new float[Model.Layers[i].InputSize];
            
            for (int j = 0; j < Model.Layers[i].OutputSize; j++)
            {
                BaseActivation? activationFunction = Model.Layers[i].ActivationFunction;
                if (activationFunction != null)
                {
                    outputGradients[j] *= activationFunction.CalculateDerivative(Model.Layers[i].RawOutputs[j]);
                }
            }
            
            // Scope Copies
            int i1 = i;
            float[] gradients = outputGradients;
            
            Parallel.For(0, Model.Layers[i].OutputSize, j =>
            {
                for (int k = 0; k < Model.Layers[i1].InputSize; k++)
                {
                    weightGradients[i1][k, j] = gradients[j] * Model.Layers[i1].Inputs[k];
                    inputGradients[k] += gradients[j] * Model.Layers[i1].Weights[k, j];
                }
                
                biasGradients[i1][j] = gradients[j]; 
            });
    
            outputGradients = inputGradients;
        }
        
        // TODO: find better solution for exploding gradients, this is a temporary solution
        for (int i = Model.Layers.Length - 1; i >= 0; i--)
        {
            for (int x = 0; x < weightGradients[i].GetLength(0); x++)
            {
                for (int y = 0; y < weightGradients[i].GetLength(1); y++)
                {
                    if (weightGradients[i][x, y] > MathF.Abs(GradientClip))
                    {
                        weightGradients[i][x, y] = MathF.Abs(GradientClip);
                    }
                    else if (weightGradients[i][x, y] < MathF.Abs(GradientClip) * -1.0F)
                    {
                        weightGradients[i][x, y] = MathF.Abs(GradientClip) * -1.0F;
                    }
                    else
                    {
                        weightGradients[i][x, y] = weightGradients[i][x, y];
                    }
                }
            }
            
            for (int j = 0; j < biasGradients[i].Length; j++)
            {
                if (biasGradients[i][j] > MathF.Abs(GradientClip))
                {
                    biasGradients[i][j] = MathF.Abs(GradientClip);
                }
                else if (biasGradients[i][j] < MathF.Abs(GradientClip) * -1.0F)
                {
                    biasGradients[i][j] = MathF.Abs(GradientClip) * -1.0F;
                }
                else
                {
                    biasGradients[i][j] = biasGradients[i][j];
                }
            }
        }
    
        return new BackpropagationResult(weightGradients, biasGradients);
    }
}