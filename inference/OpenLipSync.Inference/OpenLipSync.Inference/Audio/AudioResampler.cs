using System;

namespace OpenLipSync.Inference.Audio;

/// <summary>
/// High-quality audio resampler for converting between sample rates.
/// Uses linear interpolation with anti-aliasing for real-time processing.
/// </summary>
public sealed class AudioResampler : IDisposable
{
    private readonly double _ratio;
    private readonly int _inputSampleRate;
    private readonly int _outputSampleRate;
    private readonly LowPassFilter _antiAliasFilter;
    
    // State for interpolation
    private double _position;
    private float _previousSample;
    private bool _disposed;

    public AudioResampler(int inputSampleRate, int outputSampleRate)
    {
        if (inputSampleRate <= 0) throw new ArgumentException("Input sample rate must be positive", nameof(inputSampleRate));
        if (outputSampleRate <= 0) throw new ArgumentException("Output sample rate must be positive", nameof(outputSampleRate));

        _inputSampleRate = inputSampleRate;
        _outputSampleRate = outputSampleRate;
        _ratio = (double)inputSampleRate / outputSampleRate;
        
        // Apply anti-aliasing filter if downsampling
        if (_ratio > 1.0)
        {
            double cutoffFreq = 0.45 * outputSampleRate; // Nyquist frequency with safety margin
            _antiAliasFilter = new LowPassFilter(inputSampleRate, cutoffFreq);
        }
        
        _position = 0.0;
        _previousSample = 0.0f;
    }

    public int InputSampleRate => _inputSampleRate;
    public int OutputSampleRate => _outputSampleRate;
    public double ResampleRatio => _ratio;

    /// <summary>
    /// Resample input audio to the target sample rate.
    /// </summary>
    /// <param name="input">Input audio samples at source sample rate</param>
    /// <returns>Resampled audio at target sample rate</returns>
    public float[] Resample(ReadOnlySpan<float> input)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(AudioResampler));
        if (input.IsEmpty) return Array.Empty<float>();

        // Apply anti-aliasing filter if downsampling
        var filteredInput = _antiAliasFilter?.Process(input) ?? input.ToArray();
        
        // Calculate output length
        int outputLength = (int)Math.Ceiling(filteredInput.Length / _ratio);
        var output = new float[outputLength];
        
        int outputIndex = 0;
        
        while (outputIndex < outputLength && _position < filteredInput.Length - 1)
        {
            // Linear interpolation between samples
            int baseIndex = (int)Math.Floor(_position);
            double fraction = _position - baseIndex;
            
            float sample1 = baseIndex >= 0 ? filteredInput[baseIndex] : _previousSample;
            float sample2 = baseIndex + 1 < filteredInput.Length ? filteredInput[baseIndex + 1] : filteredInput[^1];
            
            output[outputIndex] = (float)(sample1 * (1.0 - fraction) + sample2 * fraction);
            
            outputIndex++;
            _position += _ratio;
        }
        
        // Update position for next call (handle continuous streaming)
        _position -= filteredInput.Length;
        if (_position < 0) _position = 0;
        
        // Remember last sample for interpolation continuity
        _previousSample = filteredInput.Length > 0 ? filteredInput[^1] : _previousSample;
        
        return output;
    }

    /// <summary>
    /// Calculate expected output length for a given input length.
    /// Useful for pre-allocating buffers.
    /// </summary>
    public int CalculateOutputLength(int inputLength)
    {
        return (int)Math.Ceiling(inputLength / _ratio);
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _antiAliasFilter?.Dispose();
            _disposed = true;
        }
    }
}

/// <summary>
/// Simple single-pole low-pass filter for anti-aliasing.
/// </summary>
internal sealed class LowPassFilter : IDisposable
{
    private readonly double _alpha;
    private double _previousOutput;

    public LowPassFilter(int sampleRate, double cutoffFrequency)
    {
        if (sampleRate <= 0) throw new ArgumentException("Sample rate must be positive", nameof(sampleRate));
        if (cutoffFrequency <= 0) throw new ArgumentException("Cutoff frequency must be positive", nameof(cutoffFrequency));

        // Calculate filter coefficient
        double rc = 1.0 / (2.0 * Math.PI * cutoffFrequency);
        double dt = 1.0 / sampleRate;
        _alpha = dt / (rc + dt);
        _previousOutput = 0.0;
    }

    public float[] Process(ReadOnlySpan<float> input)
    {
        var output = new float[input.Length];
        
        for (int i = 0; i < input.Length; i++)
        {
            _previousOutput += _alpha * (input[i] - _previousOutput);
            output[i] = (float)_previousOutput;
        }
        
        return output;
    }

    public void Dispose()
    {
        // No resources to dispose
    }
}
