using SharpTorch;
using SharpTorch.Losses;
using SharpTorch.Optimizers;
using SharpTorchSamples;

int inputSize = 10;
float[,] xData = new float[inputSize, 1];
float[,] yData = new float[inputSize, 1];

for (int i = 0; i < inputSize; i++)
{
    xData[i, 0] = i;
    yData[i, 0] = i * i;
}

TestModel model = new();
Trainer trainer = new(model, new MeanSquaredError(), xData, yData, 1e-4f, null, 1, 75000, 250);
model.Train();
trainer.Train();

model.DebugPrint();
Console.WriteLine();
model.Eval();

for (int i = 0; i < xData.GetLength(0); i++)
{
    float[] output = model.Forward([xData[i, 0]]);
    Console.WriteLine($"Input: {xData[i, 0]}, Output: {output[0]}, Expected output: {yData[i, 0]}");
}

Console.ReadLine();
