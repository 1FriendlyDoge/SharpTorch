// using System.Drawing;
// using SharpTorch;
// using SharpTorchSamples;
//
// string baseDir = AppDomain.CurrentDomain.BaseDirectory;
// string dataPath = Path.Combine(baseDir, "Data");
//
// byte[,,] imagesRaw = UByteReader.ReadImagesFromIdx3Ubyte(Path.Combine(dataPath, "train-images.idx3-ubyte"));
// int[] labelsRaw = UByteReader.ReadLabelsFromIdx1Ubyte(Path.Combine(dataPath, "train-labels.idx1-ubyte"));
//
// float[,,] images = UByteReader.ConvertToFloats(imagesRaw);
// float[,] projImage = Utils.Project2D(images, 0);


using SharpTorch;
using SharpTorch.Losses;
using SharpTorchSamples;

int inputSize = 500;
float[,] xData = new float[inputSize, 1];
float[,] yData = new float[inputSize, 1];

for (int i = 0; i < inputSize; i++)
{
    xData[i, 0] = i;
    yData[i, 0] = i * 2;
}

TestModel model = new();
Trainer trainer = new(model, new MeanSquaredError(), xData, yData, 1e-6f, 100, 1000000);
model.Train();
trainer.Train();

model.Eval();
float[] testInput = [5];
float[] testOutput = model.Forward(testInput);

Console.WriteLine($"Test input: {testInput[0]}, Test output: {testOutput[0]}, Expected output: {testInput[0] * 2}");