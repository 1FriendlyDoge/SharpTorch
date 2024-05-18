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
    
    public Trainer(BaseModel model, BaseLoss loss, float[,] x, float[,] y, float learningRate = 0.01f, int batchSize = 1, int epochs = 10)
    {
        Model = model;
        Loss = loss;
        LearningRate = learningRate;
        BatchSize = batchSize;
        Epochs = epochs;
        X = x;
        Y = y;
    }
    
    public void Train(CancellationTokenSource? cts = null)
    {
        cts ??= new CancellationTokenSource();
        
        Model.Train();
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
                for (int batchIndex = 0; batchIndex < BatchSize; batchIndex++)
                {
                    if (dataIndex + batchIndex >= X.Length)
                    {
                        break;
                    }
                    
                    int index = dataIndex + batchIndex;

                    int localEpoch = epoch;
                    tasks[batchIndex] = Task.Run(() =>
                    {
                        float[] xData = Utils.Project1D(X, index);
                        float[] yResult = Utils.Project1D(Y, index);
                        
                        float[] yPredicted = Model.Forward(xData);
                        
                        float loss = Loss.CalculateAll(yPredicted, yResult);
                        BackpropagationResult backpropagationResult = Backward(yPredicted, yResult, Loss);
                        Model.UpdateValues(backpropagationResult, LearningRate);
                        
                        Console.WriteLine($"Epoch: {localEpoch + 1}, Loss: {loss}");
                    }, cts.Token);
                }
                
                Task.WaitAll(tasks);
            }
        }
    }
    
    public BackpropagationResult Backward(float[] yPredicted, float[] yResult, BaseLoss loss)
    {
        List<float[,]> weightGradients = [];
        List<float[]> biasGradients = [];
        
        for (int i = 0; i < Model.Layers.Length; i++)
        {
            weightGradients.Add(new float[Model.Layers[i].InputSize, Model.Layers[i].OutputSize]);
            biasGradients.Add(new float[Model.Layers[i].OutputSize]);
        }
        
        float[] outputGradients = loss.GradientAll(yPredicted, yResult);

        for (int i = Model.Layers.Length - 1; i >= 0; i--)
        {
            float[] inputGradients = new float[Model.Layers[i].InputSize];

            for (int j = 0; j < Model.Layers[i].OutputSize; j++)
            {
                for (int k = 0; k < Model.Layers[i].InputSize; k++)
                {
                    weightGradients[i][k, j] = outputGradients[j] * Model.Layers[i].Inputs[k];
                    inputGradients[k] += outputGradients[j] * Model.Layers[i].Weights[k, j];
                }

                biasGradients[i][j] = outputGradients[j];
            }

            outputGradients = inputGradients;
        }

        return new BackpropagationResult(weightGradients, biasGradients);
    }
}