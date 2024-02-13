using SharpTorch.Losses;
using SharpTorch.Models;

namespace SharpTorch;

public class Trainer<TModel, TLoss> where TModel : ModelBase where TLoss : BaseLoss
{
    TModel Model { get; set; }
    TLoss Loss { get; set; }
    float LearningRate { get; set; }
    int BatchSize { get; set; }
    int Epochs { get; set; }
    float[] X { get; set; }
    float[] Y { get; set; }
    
    public Trainer(TModel model, TLoss loss, float[] x, float[] y, float learningRate = 0.01f, int batchSize = 32, int epochs = 10)
    {
        Model = model;
        Loss = loss;
        LearningRate = learningRate;
        BatchSize = batchSize;
        Epochs = epochs;
        X = x;
        Y = y;
    }
    
    public async Task Train(CancellationToken cts)
    {
        for (int epoch = 0; epoch < Epochs; epoch++)
        {
            if(cts.IsCancellationRequested)
            {
                return;
            }

            await Parallel.ForAsync(0, MathF.Ceiling(X.Length / BatchSize), cts, x =>
            {

            });
        }
    }
    
    private float[] Backward(float[] yHat, float[] y)
    {
        float[] gradients = new float[yHat.Length];
        
        for (int i = 0; i < yHat.Length; i++)
        {
            gradients[i] = 2 * (yHat[i] - y[i]);
        }
        
        return gradients;
    }
}