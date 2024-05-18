using SharpTorch.Layers;
using SharpTorch.Losses;
using SharpTorch.Models;
using SharpTorch.Structs;

namespace SharpTorch;

public class Trainer
{
    private BaseModel Model { get; set; }
    private BaseLoss Loss { get; set; }
    private float LearningRate { get; set; }
    private int BatchSize { get; set; }
    private int Epochs { get; set; }
    private float[,] X { get; set; }
    private float[,] Y { get; set; }
    private int ValidationInterval { get; set; }
    
    public Trainer(BaseModel model, BaseLoss loss, float[,] x, float[,] y, float learningRate = 0.01f, int batchSize = 1, int epochs = 10, int validationInterval = 5)
    {
        Model = model;
        Loss = loss;
        LearningRate = learningRate;
        BatchSize = batchSize;
        Epochs = epochs;
        X = x;
        Y = y;
        ValidationInterval = validationInterval;
    }
    
    public void Train(CancellationTokenSource? cts = null)
    {
        cts ??= new CancellationTokenSource();
        Console.WriteLine("Training...");
        for (int epoch = 0; epoch < Epochs; epoch++)
        {
            for (int dataIndex = 0; dataIndex < X.Length; dataIndex += BatchSize)
            {
                if (cts is {IsCancellationRequested: true})
                {
                    return;
                }
                
                Task[] tasks = new Task[BatchSize];
                BackpropagationResult backpropagationResult = new(Model);
                for (int batchIndex = 0; batchIndex < BatchSize; batchIndex++)
                {
                    if (dataIndex + batchIndex >= X.Length)
                    {
                        break;
                    }
                    
                    int index = dataIndex + batchIndex;
                    int localEpoch = epoch;
                    var localBatchIndex = batchIndex;
                    
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
                Task.WaitAll(tasks);
                backpropagationResult.Average(BatchSize);
                Model.UpdateValues(backpropagationResult, LearningRate);
            }
            
            if (epoch % ValidationInterval == 0)
            {
                PrintValidation(epoch);
            }
        }
    }

    private void PrintValidation(int epoch)
    {
        Model.Eval();
        float totalLoss = 0;
        Parallel.For(0, X.Length, i =>
        {
            float[] xData = Utils.Project1D(X, i);
            float[] yResult = Utils.Project1D(Y, i);
            
            float[] yPredicted = Model.Forward(xData);
            totalLoss += Loss.CalculateAll(yPredicted, yResult);
        });
        Console.WriteLine($"Epoch: {epoch}, Validation Loss: {totalLoss / X.Length}");
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

        return new BackpropagationResult(weightGradients, biasGradients);
    }
}