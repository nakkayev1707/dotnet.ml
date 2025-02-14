using dotnet.ml.QA;
using Microsoft.ML;

static class Program
{
    public static void Main()
    {
        List<string> tutorialTexts = TutorialQA.LoadTutorials();
        ITransformer model = TutorialQA.TrainModel(tutorialTexts);
        
        Console.WriteLine("Ask a question:");
        string question = Console.ReadLine();
        string answer = TutorialQA.FindBestMatch(question, tutorialTexts, model);
        Console.WriteLine($"Best Answer: {answer}");
    }
}