using System.Text.Json.Serialization;

namespace OpenLipSync.Inference;

public class TrainingInfo
{
    [JsonPropertyName("multi_label")] public bool MultiLabel { get; set; }
}