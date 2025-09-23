namespace OpenLipSync.Inference.OVRCompat;

public interface IOvrLipSyncBackend : IDisposable
{
    Result Initialize(int sampleRate, int bufferSize);
    void Shutdown();

    Result CreateContext(ref uint context, ContextProviders provider, int sampleRate = 0, bool enableAcceleration = false);
    Result CreateContextWithModelFile(ref uint context, ContextProviders provider, string modelPath, int sampleRate = 0, bool enableAcceleration = false);
    Result DestroyContext(uint context);
    Result ResetContext(uint context);

    Result SendSignal(uint context, Signals signal, int arg1);

    Result ProcessFrameFloat(uint context, ReadOnlySpan<float> audio, bool stereo, ref Frame frame);
    Result ProcessFrameShort(uint context, ReadOnlySpan<short> audio, bool stereo, ref Frame frame);
}