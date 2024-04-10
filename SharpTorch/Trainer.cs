using SharpTorch.Losses;
using SharpTorch.Models;

namespace SharpTorch;

public class Trainer<TModel>(TModel model, BaseLoss loss, float[] x, float[] y, float learningRate = 0.01f, int batchSize = 32, int epochs = 10) where TModel : ModelBase<TModel>
{
    private readonly CancellationTokenSource cts = new();

    private TModel Model { get; set; } = model;
    private BaseLoss Loss { get; set; } = loss;
    private float LearningRate { get; set; } = learningRate;
    private int BatchSize { get; set; } = batchSize;
    private int Epochs { get; set; } = epochs;
    private float[] X { get; set; } = x;
    private float[] Y { get; set; } = y;

    public Task Train()
    {
        for (int epoch = 0; epoch < Epochs; epoch++)
        {
            if(cts.IsCancellationRequested)
            {
                return Task.CompletedTask;
            }
        }

        // TODO: Hardware accelerated loop

        return Task.CompletedTask;
    }

    public async Task Terminate()
    {
        await cts.CancelAsync();
    }
}