namespace OpenLipSync.Inference;

/// <summary>
/// Model configuration loaded from config.json
/// </summary>
public class ModelConfig
{
    public ModelInfo? Model { get; set; }
    public AudioInfo? Audio { get; set; }
    public TrainingInfo? Training { get; set; }
}