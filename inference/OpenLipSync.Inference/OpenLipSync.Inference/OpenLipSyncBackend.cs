using System;
using System.Collections.Concurrent;
using System.IO;
using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenLipSync.Inference.Audio;
using OpenLipSync.Inference.OVRCompat;

namespace OpenLipSync.Inference;

/// <summary>
/// OpenLipSync backend implementation that processes audio through a trained TCN model
/// to generate real-time viseme predictions compatible with OVR LipSync interface.
/// </summary>
public sealed class OpenLipSyncBackend : IOvrLipSyncBackend
{
    private const int VISEME_COUNT = Frame.VisemeCount;
    private const float SILENCE_THRESHOLD = 0.001f;
    
    private readonly object _lock = new();
    private readonly ConcurrentDictionary<uint, AudioContext> _contexts = new();
    private InferenceSession? _onnxSession;
    private ModelConfig? _modelConfig;
    private AudioProcessingConfig? _audioConfig;
    private AudioResampler? _resampler;
    private uint _nextContextId = 1;
    private int _inputSampleRate;
    private bool _initialized;
    private bool _disposed;

    public bool IsInitialized => _initialized;
    public int SampleRate => _inputSampleRate;

    /// <summary>
    /// Initialize the backend with the specified audio parameters.
    /// Automatically loads the ONNX model and configuration.
    /// </summary>
    public Result Initialize(int sampleRate, int bufferSize)
    {
        if (_disposed) return Result.Unknown;
        if (_initialized) return Result.Success;

        try
        {
            _inputSampleRate = sampleRate;

            // Load ONNX model and configuration
            if (!LoadModel())
            {
                return Result.MissingDLL; // Model file not found
            }

            // Create resampler if needed (use actual target sample rate from config)
            int targetSampleRate = _audioConfig?.SampleRate ?? 16000; // fallback to 16kHz if config not loaded
            if (sampleRate != targetSampleRate)
            {
                _resampler = new AudioResampler(sampleRate, targetSampleRate);
            }

            _initialized = true;
            return Result.Success;
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"OpenLipSync initialization failed: {ex.Message}");
            return Result.CannotCreateContext;
        }
    }

    public void Shutdown()
    {
        if (!_initialized) return;

        lock (_lock)
        {
            // Dispose all contexts
            foreach (var context in _contexts.Values)
            {
                context.Dispose();
            }
            _contexts.Clear();

            // Dispose model and resampler
            _onnxSession?.Dispose();
            _onnxSession = null;
            _resampler?.Dispose();
            _resampler = null;

            _initialized = false;
        }
    }

    public Result CreateContext(ref uint context, ContextProviders provider, int sampleRate = 0, bool enableAcceleration = false)
    {
        if (!_initialized || _audioConfig == null) return Result.CannotCreateContext;

        try
        {
            var audioContext = new AudioContext(_inputSampleRate, _resampler, _audioConfig);
            context = _nextContextId++;
            _contexts[context] = audioContext;
            
            return Result.Success;
        }
        catch
        {
            return Result.CannotCreateContext;
        }
    }

    public Result CreateContextWithModelFile(ref uint context, ContextProviders provider, string modelPath, int sampleRate = 0, bool enableAcceleration = false)
    {
        // For now, use the same model for all contexts
        // In future, could support loading different models per context
        return CreateContext(ref context, provider, sampleRate, enableAcceleration);
    }

    public Result DestroyContext(uint context)
    {
        if (!_initialized) return Result.Unknown;

        if (_contexts.TryRemove(context, out var audioContext))
        {
            audioContext.Dispose();
            return Result.Success;
        }

        return Result.InvalidParam;
    }

    public Result ResetContext(uint context)
    {
        if (!_initialized) return Result.Unknown;

        if (_contexts.TryGetValue(context, out var audioContext))
        {
            audioContext.Reset();
            return Result.Success;
        }

        return Result.InvalidParam;
    }

    public Result SendSignal(uint context, Signals signal, int arg1, int arg2)
    {
        if (!_initialized) return Result.Unknown;

        if (_contexts.TryGetValue(context, out var audioContext))
        {
            return audioContext.SendSignal(signal, arg1, arg2);
        }

        return Result.InvalidParam;
    }

    public Result ProcessFrameFloat(uint context, ReadOnlySpan<float> audio, bool stereo, ref Frame frame)
    {
        if (!_initialized || _onnxSession == null) return Result.Unknown;

        if (!_contexts.TryGetValue(context, out var audioContext))
        {
            return Result.InvalidParam;
        }

        try
        {
            // Convert stereo to mono if needed
            var monoAudio = stereo ? ConvertStereoToMono(audio) : audio.ToArray();

            // Process audio through context
            audioContext.ProcessAudio(monoAudio);

            // Try to get mel features and run inference
            while (audioContext.TryGetNextMelFrame(out var melFeatures))
            {
                var visemeProbs = RunInference(melFeatures);
                audioContext.UpdateLatestResults(visemeProbs);
            }

            // Update frame with latest results
            audioContext.UpdateFrame(ref frame);

            return Result.Success;
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"ProcessFrameFloat error: {ex.Message}");
            return Result.Unknown;
        }
    }

    public Result ProcessFrameShort(uint context, ReadOnlySpan<short> audio, bool stereo, ref Frame frame)
    {
        if (!_initialized) return Result.Unknown;

        // Convert short samples to float
        var floatAudio = new float[audio.Length];
        for (int i = 0; i < audio.Length; i++)
        {
            floatAudio[i] = audio[i] / 32768f;
        }

        return ProcessFrameFloat(context, floatAudio, stereo, ref frame);
    }

    private bool LoadModel()
    {
        try
        {
            // Look for model files in standard locations
            var modelPaths = new[]
            {
                "model.onnx",
                "model/model.onnx",
                "export/model.onnx",

                #if DEBUG //Test model paths for debugging
                Path.Combine("export", "tcn_new_cf_implementation_2025-08-16_12-49-05", "model.onnx"),
                Path.Combine("..", "..", "..", "..", "..", "export", "tcn_new_cf_implementation_2025-08-16_12-49-05", "model.onnx"),
                "/home/kyuubi/Repos/OpenLipSync/export/tcn_new_cf_implementation_2025-08-16_12-49-05/model.onnx"
                #endif
            };

            string? modelPath = null;
            string? configPath = null;

            foreach (var path in modelPaths)
            {
                if (File.Exists(path))
                {
                    modelPath = path;
                    var dir = Path.GetDirectoryName(path) ?? "";
                    var potentialConfigPath = Path.Combine(dir, "config.json");
                    if (File.Exists(potentialConfigPath))
                    {
                        configPath = potentialConfigPath;
                    }
                    break;
                }
            }

            if (modelPath == null)
            {
                System.Diagnostics.Debug.WriteLine("ONNX model file not found");
                return false;
            }

            // Load model configuration
            if (configPath != null)
            {
                var configJson = File.ReadAllText(configPath);
                _modelConfig = JsonSerializer.Deserialize<ModelConfig>(configJson);
                
                if (_modelConfig != null)
                {
                    _audioConfig = AudioProcessingConfig.FromModelConfig(_modelConfig);
                    
                    System.Diagnostics.Debug.WriteLine($"Loaded config: {_audioConfig.SampleRate}Hz, {_audioConfig.NMels} mels, {_audioConfig.Fps}fps, {_audioConfig.HopLengthSamples} hop samples");
                    
                    // Log the sample rate from config
                    System.Diagnostics.Debug.WriteLine($"Model expects {_audioConfig.SampleRate}Hz (from config)");
                }
            }

            // Create ONNX session with CPU provider (for compatibility)
            var sessionOptions = new SessionOptions();
            sessionOptions.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING;
            
            _onnxSession = new InferenceSession(modelPath, sessionOptions);

            System.Diagnostics.Debug.WriteLine($"Loaded OpenLipSync model: {modelPath}");
            return true;
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Failed to load model: {ex.Message}");
            return false;
        }
    }

    private float[] RunInference(float[] melFeatures)
    {
        if (_onnxSession == null || _audioConfig == null) return new float[Frame.VisemeCount];

        try
        {
            // Prepare input tensor: [batch=1, time=1, features=n_mels]
            var inputTensor = new DenseTensor<float>(new[] { 1, 1, melFeatures.Length });
            for (int i = 0; i < melFeatures.Length; i++)
            {
                inputTensor[0, 0, i] = melFeatures[i];
            }

            // Create input dictionary
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("audio_features", inputTensor)
            };

            // Run inference
            using var results = _onnxSession.Run(inputs);
            
            // Extract output tensor: [batch=1, time=1, classes=num_visemes]
            var outputTensor = results.First().AsTensor<float>();
            
            // Get number of visemes from model config
            int numVisemes = _modelConfig?.model?.num_visemes ?? Frame.VisemeCount;
            
            // Convert logits to probabilities using softmax
            var logits = new float[numVisemes];
            for (int i = 0; i < numVisemes; i++)
            {
                logits[i] = outputTensor[0, 0, i];
            }

            return Softmax(logits);
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Inference error: {ex.Message}");
            return new float[VISEME_COUNT];
        }
    }

    private static float[] Softmax(float[] logits)
    {
        var probs = new float[logits.Length];
        float maxLogit = float.MinValue;
        
        // Find max for numerical stability
        for (int i = 0; i < logits.Length; i++)
        {
            if (logits[i] > maxLogit) maxLogit = logits[i];
        }

        // Compute exp and sum
        float sum = 0f;
        for (int i = 0; i < logits.Length; i++)
        {
            probs[i] = MathF.Exp(logits[i] - maxLogit);
            sum += probs[i];
        }

        // Normalize
        if (sum > 0f)
        {
            for (int i = 0; i < probs.Length; i++)
            {
                probs[i] /= sum;
            }
        }

        return probs;
    }

    private static float[] ConvertStereoToMono(ReadOnlySpan<float> stereoAudio)
    {
        var monoAudio = new float[stereoAudio.Length / 2];
        for (int i = 0; i < monoAudio.Length; i++)
        {
            monoAudio[i] = (stereoAudio[i * 2] + stereoAudio[i * 2 + 1]) * 0.5f;
        }
        return monoAudio;
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            Shutdown();
            _disposed = true;
        }
    }
}

/// <summary>
/// Audio processing context for a single audio stream.
/// Handles resampling, buffering, and mel spectrogram computation.
/// </summary>
internal sealed class AudioContext : IDisposable
{
    private readonly AudioRingBuffer _ringBuffer;
    private readonly MelSpectrogramProcessor _melProcessor;
    private readonly AudioResampler? _resampler;
    private readonly float[] _latestVisemeResults;
    private float _smoothing = 0.7f;
    private int _frameNumber;
    private bool _disposed;

    public AudioContext(int inputSampleRate, AudioResampler? resampler, AudioProcessingConfig audioConfig)
    {
        _ringBuffer = new AudioRingBuffer(audioConfig.SampleRate * 3); // 3 seconds at target sample rate
        _melProcessor = new MelSpectrogramProcessor(audioConfig);
        _resampler = resampler;
        _latestVisemeResults = new float[Frame.VisemeCount];
        _frameNumber = 0;
    }

    public void ProcessAudio(float[] audioSamples)
    {
        if (_disposed) return;

        // Resample if needed
        var targetSamples = _resampler?.Resample(audioSamples) ?? audioSamples;
        
        // Add to ring buffer
        _ringBuffer.Write(targetSamples);
    }

    public bool TryGetNextMelFrame(out float[] melFeatures)
    {
        return _melProcessor.TryProcessNextHop(_ringBuffer, out melFeatures);
    }

    public void UpdateLatestResults(float[] visemeProbs)
    {
        if (_disposed || visemeProbs.Length != Frame.VisemeCount) return;

        // Apply smoothing
        for (int i = 0; i < Frame.VisemeCount; i++)
        {
            _latestVisemeResults[i] = _latestVisemeResults[i] * _smoothing + visemeProbs[i] * (1f - _smoothing);
        }
    }

    public void UpdateFrame(ref Frame frame)
    {
        if (_disposed) return;

        frame.frameNumber = ++_frameNumber;
        frame.frameDelay = 0;
        
        Array.Copy(_latestVisemeResults, frame.Visemes, Math.Min(_latestVisemeResults.Length, frame.Visemes.Length));
        
        // Set laughter score to 0 for now (could be enhanced with laughter detection)
        frame.laughterScore = 0f;
    }

    public Result SendSignal(Signals signal, int arg1, int arg2)
    {
        if (_disposed) return Result.Unknown;

        switch (signal)
        {
            case Signals.VisemeSmoothing:
                _smoothing = Math.Clamp(arg1 / 100f, 0f, 1f);
                return Result.Success;
            
            default:
                return Result.Success; // Ignore unsupported signals
        }
    }

    public void Reset()
    {
        if (_disposed) return;

        _ringBuffer.Clear();
        Array.Clear(_latestVisemeResults);
        _frameNumber = 0;
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _ringBuffer?.Dispose();
            _melProcessor?.Dispose();
            _disposed = true;
        }
    }
}

/// <summary>
/// Model configuration loaded from config.json
/// </summary>
public class ModelConfig
{
    public ModelInfo? model { get; set; }
    public AudioInfo? audio { get; set; }
}

public class ModelInfo
{
    public int num_visemes { get; set; }
    public string? name { get; set; }
}

public class AudioInfo
{
    public int sample_rate { get; set; }
    public int hop_length_ms { get; set; }
    public int window_length_ms { get; set; }
    public int n_mels { get; set; }
    public float fmin { get; set; }
    public float fmax { get; set; }
    public int n_fft { get; set; }
    public string? normalization { get; set; }
    public float fps { get; set; }
}
