using SharpTorch;
using SharpTorch.Layers;
using SharpTorch.Losses;

namespace SharpTorchSamples;

public static class DebugTest
{
    public static void Run()
    {
        int inputSize = 8;
        float[,] xData = new float[inputSize, 1];
        float[,] yData = new float[inputSize, 1];

        for (int i = 0; i < inputSize; i++)
        {
            xData[i, 0] = i;
            yData[i, 0] = i * i;
        }

        TestModel model = new();
        Trainer trainer = new(model, new MeanSquaredError(), xData, yData, 1e-4f, 1, 50000, 500);
        model.Train();
        trainer.Train();

        model.Eval();
        float[] testInput = [4];
        float[] testOutput = model.Forward(testInput);

        Console.WriteLine($"Test input: {testInput[0]}, Test output: {testOutput[0]}, Expected output: {testInput[0] * testInput[0]}");

        model.DebugPrint();
    }
}