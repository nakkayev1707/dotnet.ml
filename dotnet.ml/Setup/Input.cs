using Microsoft.ML.Data;

namespace dotnet.ml.Setup;

public class Input
{
    [LoadColumn(0)]
    public string Text { get; set; }

    [LoadColumn(1)]
    public bool Label { get; set; }
}