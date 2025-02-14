using Microsoft.ML;
using Microsoft.ML.Data;

namespace dotnet.ml.QA;

public class TutorialQA
{
    static readonly string DataFolder = "D:\\MY-FILES\\Development\\.NET\\dotnet.ml\\dotnet.ml\\Data";
    static readonly MLContext mlContext = new();
    
    public static List<string> LoadTutorials()
    {
        if (!Directory.Exists(DataFolder)) Directory.CreateDirectory(DataFolder);
        var files = Directory.GetFiles(DataFolder, "*.txt");
        
        var sentences = new List<string>();
        foreach (var file in files)
        {
            var content = File.ReadAllText(file);
            var splitSentences = content.Split(new[] { '.', '?', '!' }, StringSplitOptions.RemoveEmptyEntries);
            sentences.AddRange(splitSentences.Select(s => s.Trim()));
        }
        return sentences;
    }

    public static ITransformer TrainModel(List<string> texts)
    {
        var data = texts.Select(t => new TutorialData { Text = t }).ToList();
        var dataView = mlContext.Data.LoadFromEnumerable(data);
        var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", "Text");
        return pipeline.Fit(dataView);
    }

    public static string FindBestMatch(string question, List<string> texts, ITransformer model)
    {
        var engine = mlContext.Model.CreatePredictionEngine<TutorialData, FeatureVector>(model);
        var questionVector = engine.Predict(new TutorialData { Text = question }).Features;

        float bestScore = float.MinValue;
        string bestMatch = "No relevant answer found.";

        for (int i = 0; i < texts.Count; i++)
        {
            var textVector = engine.Predict(new TutorialData { Text = texts[i] }).Features;
            float similarity = CosineSimilarity(questionVector, textVector);
            if (similarity > bestScore)
            {
                bestScore = similarity;
                bestMatch = texts[i];
            }
        }
        return bestMatch;
    }

    static float CosineSimilarity(float[] v1, float[] v2)
    {
        float dot = 0, mag1 = 0, mag2 = 0;
        for (int i = 0; i < v1.Length; i++)
        {
            dot += v1[i] * v2[i];
            mag1 += v1[i] * v1[i];
            mag2 += v2[i] * v2[i];
        }
        return dot / (float)(Math.Sqrt(mag1) * Math.Sqrt(mag2));
    }
    
    class TutorialData
    {
        public string Text { get; set; }
    }

    class FeatureVector
    {
        [VectorType]
        public float[] Features { get; set; }
    }
}