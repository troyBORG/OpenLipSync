namespace OpenLipSync.Inference.OVRCompat;

public sealed class OVRLipSyncInterface : IDisposable
{
    public static readonly int VisemeCount = Frame.VisemeCount;
    public static readonly int SignalCount = Enum.GetNames(typeof(Signals)).Length;

    public bool IsInitialized => _initResult == Result.Success;
    public int SampleRate { get; private set; }
    public int BufferSize { get; private set; }

    private readonly IOvrLipSyncBackend _backend;
    private Result _initResult = Result.Unknown;

    public OVRLipSyncInterface(IOvrLipSyncBackend backend, int sampleRate, int frameSize)
    {
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        SampleRate = sampleRate;
        BufferSize = frameSize;
        _initResult = _backend.Initialize(sampleRate, frameSize);
    }

    public string? GetLastError() => _backend.LastError;

    public void Dispose()
    {
        _backend.Shutdown();
        _backend.Dispose();
        _initResult = Result.Unknown;
    }

    // Context management
    public Result CreateContext(ref uint context, ContextProviders provider, int sampleRate = 0, bool enableAcceleration = false)
        => IsInitialized ? _backend.CreateContext(ref context, provider, sampleRate, enableAcceleration) : Result.CannotCreateContext;

    public Result CreateContextWithModelFile(ref uint context, ContextProviders provider, string modelPath, int sampleRate = 0, bool enableAcceleration = false)
        => IsInitialized ? _backend.CreateContextWithModelFile(ref context, provider, modelPath, sampleRate, enableAcceleration) : Result.CannotCreateContext;

    public Result DestroyContext(uint context)
        => IsInitialized ? _backend.DestroyContext(context) : Result.Unknown;

    public Result ResetContext(uint context)
        => IsInitialized ? _backend.ResetContext(context) : Result.Unknown;

    // Control
    public Result SendSignal(uint context, Signals signal, int arg1)
        => IsInitialized ? _backend.SendSignal(context, signal, arg1) : Result.Unknown;

    // Processing (float)
    public Result ProcessFrame(uint context, float[] audioBuffer, Frame frame, bool stereo = true)
    {
        if (!IsInitialized) return Result.Unknown;
        if (audioBuffer == null || frame == null) return Result.InvalidParam;
        return _backend.ProcessFrameFloat(context, audioBuffer, stereo, ref frame);
    }

    // Processing (short)
    public Result ProcessFrame(uint context, short[] audioBuffer, Frame frame, bool stereo = true)
    {
        if (!IsInitialized) return Result.Unknown;
        if (audioBuffer == null || frame == null) return Result.InvalidParam;
        return _backend.ProcessFrameShort(context, audioBuffer, stereo, ref frame);
    }
}