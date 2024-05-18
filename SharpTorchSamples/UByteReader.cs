using System.Drawing;

namespace SharpTorchSamples;

public class UByteReader
{
    public static byte[,,] ReadImagesFromIdx3Ubyte(string filePath)
    {
        using var fileStream = new FileStream(filePath, FileMode.Open);
        using var reader = new BinaryReader(fileStream);

        // UNUSED
        reader.ReadBytes(4);
        
        int numberOfImages = ReadBigEndianInt32(reader);
        int height = ReadBigEndianInt32(reader);
        int width = ReadBigEndianInt32(reader);
        
        byte[,,] images = new byte[numberOfImages, height, width];
        
        for (int i = 0; i < numberOfImages; i++)
        {
            for (int j = 0; j < height; j++)
            {
                for (int k = 0; k < width; k++)
                {
                    images[i, j, k] = reader.ReadByte();
                }
            }
        }

        return images;
    }
    
    public static int[] ReadLabelsFromIdx1Ubyte(string filePath)
    {
        using var fileStream = new FileStream(filePath, FileMode.Open);
        using var reader = new BinaryReader(fileStream);

        // UNUSED
        reader.ReadBytes(4);

        int numberOfLabels = ReadBigEndianInt32(reader);

        int[] labels = new int[numberOfLabels];

        for (int i = 0; i < numberOfLabels; i++)
        {
            labels[i] = reader.ReadByte();
        }

        return labels;
    }
    
    public static float[,,] ConvertToGrayscaleFloats(byte[,,] images)
    {
        int numberOfImages = images.GetLength(0);
        int height = images.GetLength(1);
        int width = images.GetLength(2);

        float[,,] grayscaleImages = new float[numberOfImages, height, width];

        for (int i = 0; i < numberOfImages; i++)
        {
            for (int j = 0; j < height; j++)
            {
                for (int k = 0; k < width; k++)
                {
                    byte colorValue = images[i, j, k];
                    grayscaleImages[i, j, k] = colorValue / 255.0f;
                }
            }
        }

        return grayscaleImages;
    }
    
    public static Bitmap[] ConvertToBitmaps(byte[,,] images)
    {
        int numberOfImages = images.GetLength(0);
        int height = images.GetLength(1);
        int width = images.GetLength(2);

        Bitmap[] bitmaps = new Bitmap[numberOfImages];

        for (int i = 0; i < numberOfImages; i++)
        {
            bitmaps[i] = new Bitmap(width, height);

            for (int j = 0; j < height; j++)
            {
                for (int k = 0; k < width; k++)
                {
                    byte colorValue = images[i, j, k];
                    Color color = Color.FromArgb(colorValue, colorValue, colorValue);
                    bitmaps[i].SetPixel(k, j, color);
                }
            }
        }

        return bitmaps;
    }
    
    public static Bitmap ConvertToBitmap(byte[,] image)
    {
        int height = image.GetLength(0);
        int width = image.GetLength(1);

        Bitmap bitmap = new Bitmap(width, height);
        
        for (int j = 0; j < height; j++)
        {
            for (int k = 0; k < width; k++)
            {
                byte colorValue = image[j, k];
                Color color = Color.FromArgb(colorValue, colorValue, colorValue);
                bitmap.SetPixel(k, j, color);
            }
        }

        return bitmap;
    }
    
    private static int ReadBigEndianInt32(BinaryReader reader)
    {
        var bytes = reader.ReadBytes(4);
        if (BitConverter.IsLittleEndian)
        {
            Array.Reverse(bytes);
        }
        return BitConverter.ToInt32(bytes, 0);
    }
}