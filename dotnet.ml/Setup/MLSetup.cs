using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

namespace dotnet.ml.Setup;

public class MLSetup
{
    private MLContext _context = new();
    
    // private ITransformer Train()
    // {
    //     IDataView trainingData = _context.Data
    //         .LoadFromTextFile<Input>("path_to_file", hasHeader: true);
    //     
    //     var pipeline = _context.Transforms.Text
    //         .FeaturizeText("Features", nameof(SentimentIssue.Text))
    //         .Append(_context.BinaryClassification.Trainers
    //             .LbfgsLogisticRegression("Label", "Features"));
    //     
    //     ITransformer trainedModel = pipeline.Fit(trainingData);
    //
    //     return trainedModel;
    // }

    public void Evaluate()
    {
        
    }

    public void Embedding()
    {
        var path = "D:\\MY-FILES\\Development\\.NET\\dotnet.ml\\dotnet.ml\\Setup\\data.csv";
        IDataView? dataView = _context.Data.LoadFromTextFile<Input>(path, hasHeader: true);

        // Define the pipeline
        var pipeline = _context.Transforms.Text
            .FeaturizeText("Features", "Text")
            .Append(_context.BinaryClassification.Trainers
                .SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));

        // Train the model
        var model = pipeline.Fit(dataView);

        // Make predictions
        var predictor = _context.Model.CreatePredictionEngine<Input, Prediction>(model);
        var prediction = predictor.Predict(new Input() { Text = "I absolutely like this product" });
        Console.WriteLine($"Predicted Label: {prediction.PredictedLabel}");
    }

    public void Predict(ITransformer trainedModel)
    {
        var predictionEngine = _context.Model
            .CreatePredictionEngine<Input, Output>(trainedModel);

        var sampleStatement = new Input() { Text = "This is a horrible movie" };

        var prediction = predictionEngine.Predict(sampleStatement);
        Console.WriteLine(prediction);
    }
}