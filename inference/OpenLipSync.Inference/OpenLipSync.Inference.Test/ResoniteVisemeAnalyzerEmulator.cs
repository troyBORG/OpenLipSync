using OpenLipSync.Inference.OVRCompat;

namespace OpenLipSync.Inference.Test;

/// <summary>
/// Minimal emulator of Resonite's VisemeAnalyzer for testing OVR-compatible backends.
/// </summary>
public sealed class ResoniteVisemeAnalyzerEmulator : IDisposable
{
    public bool Enabled { get; set; } = true;
    public float Smoothing { get; set; } = 0.7f;

    private readonly IOvrLipSyncBackend _backend;
    private readonly OVRLipSyncInterface _ovrLipSync;
    private OVRLipSyncContext? _analysisContext;
    private float[]? _buffer;
    private readonly float[] _analysis = new float[16];

    public bool IsInitialized => _ovrLipSync.IsInitialized && _analysisContext?.IsInitialized == true;

    public ResoniteVisemeAnalyzerEmulator(IOvrLipSyncBackend backend, int sampleRate, int frameSize)
    {
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        _ovrLipSync = new OVRLipSyncInterface(backend, sampleRate, frameSize);
        
        OnAwake();
    }

    private void OnAwake()
    {
        Array.Clear(_analysis, 0, _analysis.Length);
        
        if (_ovrLipSync.IsInitialized)
        {
            _analysisContext = new OVRLipSyncContext(_ovrLipSync);
        }
    }

    public void OnAudioUpdate(ReadOnlySpan<float> audioData, int frameSize)
    {
        if (!Enabled)
        {
            Array.Clear(_analysis, 0, _analysis.Length);
            return;
        }

        if (audioData.Length > 0)
        {
            if (_buffer == null || _buffer.Length != frameSize)
            {
                _buffer = new float[frameSize];
            }

            Array.Clear(_buffer, 0, _buffer.Length);
            int copyCount = Math.Min(audioData.Length, frameSize);
            audioData.Slice(0, copyCount).CopyTo(_buffer);

            if (_analysisContext != null)
            {
                _analysisContext.Update(Smoothing);
                _analysisContext.Analyze(_buffer, _analysis, null);
            }
        }
        else
        {
            Array.Clear(_analysis, 0, _analysis.Length);
        }
    }

    public float this[int visemeIndex] => visemeIndex >= 0 && visemeIndex < _analysis.Length ? _analysis[visemeIndex] : 0f;

    public void Dispose()
    {
        _analysisContext?.Dispose();
        _analysisContext = null;
        _ovrLipSync?.Dispose();
        _backend?.Dispose();
    }
}
