using System.Text.Json.Serialization;

namespace OpenLipSync.Inference;

public class AudioInfo
{
    [JsonPropertyName("sample_rate")] public int SampleRate { get; set; }
    [JsonPropertyName("hop_length_ms")] public int HopLengthMs { get; set; }
    [JsonPropertyName("window_length_ms")] public int WindowLengthMs { get; set; }
    [JsonPropertyName("n_mels")] public int NMels { get; set; }
    [JsonPropertyName("fmin")] public float Fmin { get; set; }
    [JsonPropertyName("fmax")] public float Fmax { get; set; }
    [JsonPropertyName("n_fft")] public int NFft { get; set; }
    [JsonPropertyName("normalization")] public string? Normalization { get; set; }
    [JsonPropertyName("fps")] public float Fps { get; set; }
}