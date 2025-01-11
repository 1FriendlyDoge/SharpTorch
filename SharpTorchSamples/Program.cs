using ScottPlot;
using SharpTorch;
using SharpTorch.ActivationFunctions;
using SharpTorch.Losses;
using SharpTorch.Models;

const int inputSize = 14;
float[,] xData = new float[inputSize * 10, 1];
float[,] yData = new float[inputSize * 10, 1];

for (int i = 1; i < inputSize * 10; i++)
{
    xData[i, 0] = i * 0.1F;
    yData[i, 0] = i * i * 0.01F;
}

LinearSequentialNetwork model = new([
    (1, 16, new ReLU()),
    (16, 16, new ReLU()),
    (16, 1, new ReLU())
]);

Trainer trainer = new(model, new MeanSquaredError(), xData, yData, 1e-5f, null, 20, 100000, 1000, gradientClip: float.MaxValue);
model.Train();
trainer.Train();

model.DebugPrint();
Console.WriteLine();
model.Eval();

double[] xs = new double[xData.GetLength(0) / 10];
double[] ys = new double[xData.GetLength(0) / 10];

for (int i = 1; i < xData.GetLength(0) / 10; i++)
{
    float[] output = model.Forward([xData[i * 10, 0]]);
    Console.WriteLine($"Input: {xData[i * 10, 0]}, Output: {output[0]}, Expected output: {yData[i * 10, 0]}");
    xs[i] = xData[i * 10, 0];
    ys[i] = output[0];
}

Plot plt = new Plot();
plt.Add.Scatter(xs, ys, Colors.Black);

double[] actualYs = new double[xData.GetLength(0) / 10];
for (int i = 1; i < xData.GetLength(0) / 10; i++)
{
    actualYs[i] = yData[i * 10, 0];
}

plt.Add.Scatter(xs, actualYs, Colors.Blue);
plt.SavePng("output.png", 1024, 1024);

Console.ReadLine();
