using System.Text.Json.Serialization;

namespace OpenLipSync.Inference;

public class ModelInfo
{
    [JsonPropertyName("num_visemes")] public int NumVisemes { get; set; }
    [JsonPropertyName("name")] public string? Name { get; set; }
}