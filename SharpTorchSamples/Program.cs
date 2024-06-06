using System.Globalization;
using CsvHelper;
using SharpTorch;
using SharpTorch.ActivationFunctions;
using SharpTorch.Losses;
using SharpTorch.Models;
using SharpTorchSamples;

List<DataPoint> emails = [];

using (var reader = new StreamReader(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "emails.csv")))
using (var csv = new CsvReader(reader, CultureInfo.InvariantCulture))
{
    emails = csv.GetRecords<DataPoint>().ToList();
}

BOWTokenizer sequenceTokenizer = new();

(string, int)[] data = new (string, int)[emails.Count];
for (int i = 0; i < emails.Count; i++)
{
    data[i] = (emails[i].Text, emails[i].Spam);
}

List<(int[], int)> tokenizedData = [];
foreach ((string input, int output) in data)
{
    int[] tokenized = sequenceTokenizer.Tokenize(input);
    tokenizedData.Add((tokenized, output));
}

int maxLen = tokenizedData.Max(x => x.Item1.Length);
LinearSequentialNetwork model = new
(
    (maxLen, 8, new ReLU()),
    (8, 1, new Sigmoid())
);

float[,] xData = new float[tokenizedData.Count, maxLen];
float[,] yData = new float[tokenizedData.Count, 1];

for (int i = 0; i < tokenizedData.Count; i++)
{
    (int[] x, int y) = tokenizedData[i];
    for (int j = 0; j < x.Length; j++)
    {
        xData[i, j] = x[j];
    }

    yData[i, 0] = y;
}

Trainer trainer = new(model, new MeanAbsoluteError(), xData, yData, 1e-3F, null, 512, 100, 1, 1, gradientClip: float.MaxValue);
trainer.Train();

model.DebugPrint();

for (int i = 0; i < /*xData.GetLength(0);*/ 100; i++)
{
    float[] output = model.Forward(Utils.Project1D(xData, i));
    Console.WriteLine($"Input: {data[i].Item1}, Output: {output[0]}, Expected output: {yData[i, 0]}");
}

string inputStr = "Kilian ist ein dumm dummer Mensch";
float[] output1 = model.Forward(SequenceTokenizer.SequenceEncode(sequenceTokenizer.Tokenize(inputStr)));
Console.WriteLine($"Input: \"{inputStr}\", Output: {output1[0]}");

// int inputSize = 14;
// float[,] xData = new float[inputSize, 1];
// float[,] yData = new float[inputSize, 1];
//
// for (int i = 0; i < inputSize; i++)
// {
//     xData[i, 0] = i;
//     yData[i, 0] = i * i;
// }
//
// TestModel model = new();
// Trainer trainer = new(model, new MeanSquaredError(), xData, yData, 1e-6f, null, 14, 250000, 1000, gradientClip: float.MaxValue);
// model.Train();
// trainer.Train();
//
// model.DebugPrint();
// Console.WriteLine();
// model.Eval();
//
// for (int i = 0; i < xData.GetLength(0); i++)
// {
//     float[] output = model.Forward([xData[i, 0]]);
//     Console.WriteLine($"Input: {xData[i, 0]}, Output: {output[0]}, Expected output: {yData[i, 0]}");
// }
//
// Console.ReadLine();
