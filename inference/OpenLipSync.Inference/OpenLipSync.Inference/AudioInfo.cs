namespace OpenLipSync.Inference;

public class AudioInfo
{
    public int SampleRate { get; set; }
    public int HopLengthMs { get; set; }
    public int WindowLengthMs { get; set; }
    public int NMels { get; set; }
    public float Fmin { get; set; }
    public float Fmax { get; set; }
    public int NFft { get; set; }
    public string? Normalization { get; set; }
    public float Fps { get; set; }
}