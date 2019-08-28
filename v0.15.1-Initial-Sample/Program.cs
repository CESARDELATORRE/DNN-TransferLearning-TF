using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Transforms;
using static Microsoft.ML.DataOperationsCatalog;
using System.Linq;
using Microsoft.ML.Data;
using Common;

namespace ImageClassification.Train
{
    public class Program
    {
        static void Main(string[] args)
        {
            string assetsRelativePath = @"../../../assets";
            string assetsPath = GetAbsolutePath(assetsRelativePath);

            var outputMlNetModelFilePath = Path.Combine(assetsPath, "outputs", "imageClassifier.zip");

            string imagesDownloadFolderPath = Path.Combine(assetsPath, "inputs", "images");

            //Download the image set and unzip
            string finalImagesFolderName = DownloadImageSet(imagesDownloadFolderPath);
            string fullImagesetFolderPath = Path.Combine(imagesDownloadFolderPath, finalImagesFolderName);

            //Only for the small dataset from Zeeshan's sample
            //string fullImagesetFolderPath = Path.Combine(imagesDownloadFolderPath, "small-imageset-original");

            // *********************************************************
            // USING ML.NET DNN 0.15.1 - Code based on Initial Sample here:
            // https://github.com/dotnet/machinelearning/blob/master/docs/samples/Microsoft.ML.Samples/Dynamic/ImageClassification/InceptionV3TransferLearning.cs
            // *********************************************************

            var mlContext = new MLContext(seed: 1);

            var data = mlContext.Data.LoadFromEnumerable(
                ImageNetData.LoadImagesFromDirectory(fullImagesetFolderPath, 1, true));

            data = mlContext.Data.ShuffleRows(data, 5);

            //Split the data 75:25 into train and test sets, train and evaluate.
            TrainTestData trainTestData = mlContext.Data.TrainTestSplit(data, testFraction: 0.25);
            IDataView trainDataView = trainTestData.TrainSet;
            IDataView testDataView = trainTestData.TestSet;

            var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label")
                .Append(mlContext.Transforms.LoadImages("ImageObject", null,
                    "ImagePath"))
                .Append(mlContext.Transforms.ResizeImages("Image",
                    inputColumnName: "ImageObject", imageWidth: 299,
                    imageHeight: 299))
                .Append(mlContext.Transforms.ExtractPixels("Image",
                    interleavePixelColors: true))
                    //scaleImage: 1 / 255f)) //This was not in the initial sample
                .Append(mlContext.Model.ImageClassification("Image",
                    "Label", arch: DnnEstimator.Architecture.InceptionV3, 
                            epoch: 20,              
                            batchSize: 10,           
                            learningRate: 0.01f      
                            ));

            var trainedModel = pipeline.Fit(trainDataView);
            var predicted = trainedModel.Transform(testDataView);
            var metrics = mlContext.MulticlassClassification.Evaluate(predicted);

            Console.WriteLine($"Micro-accuracy: {metrics.MicroAccuracy}," +
                $"macro-accuracy = {metrics.MacroAccuracy}");

            // Create prediction function and test prediction
            var predictFunction = mlContext.Model
                .CreatePredictionEngine<ImageNetData, ImagePrediction>(trainedModel);

            var prediction = predictFunction
                .Predict(ImageNetData.LoadImagesFromDirectory(fullImagesetFolderPath, 1)
                .First());

            Console.WriteLine($"Scores : [{string.Join(",", prediction.Score)}], " +
                $"Predicted Label : {prediction.PredictedLabel}");

            Console.WriteLine("Press any key to finish");
            Console.ReadKey();
        }


        public static string DownloadImageSet(string imagesDownloadFolder)
        {
            // get a set of images to teach the network about the new classes

            //SINGLE SMALL FLOWERS IMAGESET (200 files)
            string fileName = "flower_photos_small_set.zip";
            string url = $"https://mlnetfilestorage.file.core.windows.net/imagesets/flower_images/flower_photos_small_set.zip?st=2019-08-07T21%3A27%3A44Z&se=2030-08-08T21%3A27%3A00Z&sp=rl&sv=2018-03-28&sr=f&sig=SZ0UBX47pXD0F1rmrOM%2BfcwbPVob8hlgFtIlN89micM%3D";
            Web.Download(url, imagesDownloadFolder, fileName);
            Compress.UnZip(Path.Join(imagesDownloadFolder, fileName), imagesDownloadFolder);

            //SINGLE FULL FLOWERS IMAGESET (3,600 files)
            //string fileName = "flower_photos.tgz";
            //string url = $"http://download.tensorflow.org/example_images/{fileName}";
            //Web.Download(url, imagesDownloadFolder, fileName);
            //Compress.ExtractTGZ(Path.Join(imagesDownloadFolder, fileName), imagesDownloadFolder);

            //SPLIT TRAIN/TEST DATASETS (FROM SMALL IMAGESET - 200 files)
            //string fileName = "flower_photos_small_set_split.zip";
            //string url = $"https://mlnetfilestorage.file.core.windows.net/imagesets/flower_images/flower_photos_small_set_split.zip?st=2019-08-23T00%3A03%3A25Z&se=2030-08-24T00%3A03%3A00Z&sp=rl&sv=2018-03-28&sr=f&sig=qROCaSGod0mCDP87xDmGCli3o8XyKUlUUimRGGVG9RE%3D";
            //Web.Download(url, imagesDownloadFolder, fileName);
            //Compress.UnZip(Path.Join(imagesDownloadFolder, fileName), imagesDownloadFolder);

            return Path.GetFileNameWithoutExtension(fileName);
        }

        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }

    }
}

public class ImageNetData
{
    [LoadColumn(0)]
    public string ImagePath;

    [LoadColumn(1)]
    public string Label;

    public static IEnumerable<ImageNetData> LoadImagesFromDirectory(
        string folder, int repeat = 1, bool useFolderNameasLabel = false)
    {
        var files = Directory.GetFiles(folder, "*",
            searchOption: SearchOption.AllDirectories);

        foreach (var file in files)
        {
            if (Path.GetExtension(file) != ".jpg")
                continue;

            var label = Path.GetFileName(file);
            if (useFolderNameasLabel)
                label = Directory.GetParent(file).Name;
            else
            {
                for (int index = 0; index < label.Length; index++)
                {
                    if (!char.IsLetter(label[index]))
                    {
                        label = label.Substring(0, index);
                        break;
                    }
                }
            }

            for (int index = 0; index < repeat; index++)
                yield return new ImageNetData()
                {
                    ImagePath = file,
                    Label = label
                };
        }
    }
}

public class ImagePrediction
{
    [ColumnName("Score")]
    public float[] Score;

    [ColumnName("PredictedLabel")]
    public Int64 PredictedLabel;
}

/* 

// ORIGINAL CODE: https://github.com/dotnet/machinelearning/blob/master/docs/samples/Microsoft.ML.Samples/Dynamic/ImageClassification/InceptionV3TransferLearning.cs

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

namespace Samples.Dynamic
{
    public static class InceptionV3TransferLearning
    {
        /// <summary>
        /// Example use of Image classification API in a ML.NET pipeline.
        /// </summary>
        public static void Example()
        {
            var mlContext = new MLContext(seed: 1);

            var imagesDataFile = Path.GetDirectoryName(
                Microsoft.ML.SamplesUtils.DatasetUtils.DownloadImages());

            var data = mlContext.Data.LoadFromEnumerable(
                ImageNetData.LoadImagesFromDirectory(imagesDataFile, 4));

            data = mlContext.Data.ShuffleRows(data, 5);
            var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label")
                .Append(mlContext.Transforms.LoadImages("ImageObject", null,
                    "ImagePath"))
                .Append(mlContext.Transforms.ResizeImages("Image",
                    inputColumnName: "ImageObject", imageWidth: 299,
                    imageHeight: 299))
                .Append(mlContext.Transforms.ExtractPixels("Image",
                    interleavePixelColors: true))
                .Append(mlContext.Model.ImageClassification("Image",
                    "Label", arch: DnnEstimator.Architecture.InceptionV3, epoch: 4,
                    batchSize: 4));

            var trainedModel = pipeline.Fit(data);
            var predicted = trainedModel.Transform(data);
            var metrics = mlContext.MulticlassClassification.Evaluate(predicted);

            Console.WriteLine($"Micro-accuracy: {metrics.MicroAccuracy}," +
                $"macro-accuracy = {metrics.MacroAccuracy}");

            // Create prediction function and test prediction
            var predictFunction = mlContext.Model
                .CreatePredictionEngine<ImageNetData, ImagePrediction>(trainedModel);

            var prediction = predictFunction
                .Predict(ImageNetData.LoadImagesFromDirectory(imagesDataFile, 4)
                .First());

            Console.WriteLine($"Scores : [{string.Join(",", prediction.Score)}], " +
                $"Predicted Label : {prediction.PredictedLabel}");

        }
    }

    public class ImageNetData
    {
        [LoadColumn(0)]
        public string ImagePath;

        [LoadColumn(1)]
        public string Label;

        public static IEnumerable<ImageNetData> LoadImagesFromDirectory(
            string folder, int repeat = 1, bool useFolderNameasLabel = false)
        {
            var files = Directory.GetFiles(folder, "*",
                searchOption: SearchOption.AllDirectories);

            foreach (var file in files)
            {
                if (Path.GetExtension(file) != ".jpg")
                    continue;

                var label = Path.GetFileName(file);
                if (useFolderNameasLabel)
                    label = Directory.GetParent(file).Name;
                else
                {
                    for (int index = 0; index < label.Length; index++)
                    {
                        if (!char.IsLetter(label[index]))
                        {
                            label = label.Substring(0, index);
                            break;
                        }
                    }
                }

                for (int index = 0; index < repeat; index++)
                    yield return new ImageNetData()
                    {
                        ImagePath = file,
                        Label = label
                    };
            }
        }
    }

    public class ImagePrediction
    {
        [ColumnName("Score")]
        public float[] Score;

        [ColumnName("PredictedLabel")]
        public Int64 PredictedLabel;
    }
}

*/
