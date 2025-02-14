namespace dotnet.ml.Setup;

public class Prediction
{
    // Predicted label for classification tasks
    public bool PredictedLabel { get; set; }

    // Confidence score or probability of the prediction
    public float Score { get; set; }
}