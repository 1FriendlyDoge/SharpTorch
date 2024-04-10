// See https://aka.ms/new-console-template for more information

using SharpTorch;
using SharpTorch.Layers;
using SharpTorch.Losses;
using SharpTorch.Models;

SquaredAbsoluteLoss squaredAbsoluteLoss = new();
TestModel m = new();

Trainer<TestModel> trainer = new(m, squaredAbsoluteLoss, [], []);
await trainer.Terminate();