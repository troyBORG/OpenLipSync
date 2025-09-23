using System.Text.Json.Serialization;

namespace OpenLipSync.Inference;

/// <summary>
/// Model configuration loaded from config.json
/// </summary>
public class ModelConfig
{
    [JsonPropertyName("model")] public ModelInfo? Model { get; set; }
    [JsonPropertyName("audio")] public AudioInfo? Audio { get; set; }
    [JsonPropertyName("training")] public TrainingInfo? Training { get; set; }
}