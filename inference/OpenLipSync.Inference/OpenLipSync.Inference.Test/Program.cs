namespace OpenLipSync.Inference.Test;

using OpenLipSync.Inference;
using OpenLipSync.Inference.Audio;
using OpenLipSync.Inference.OVRCompat;

class Program
{
    static void Main(string[] args)
    {
        var sample = SampleLocator.FindAnySample();

        if (sample is null)
        {
            Console.WriteLine("No sample found under training/data/prepared/dev-clean.");
            return;
        }

        Console.WriteLine("Selected sample:");
        Console.WriteLine($"  WAV:   {sample.WavPath}");
        Console.WriteLine($"  LAB:   {sample.LabPath}");
        Console.WriteLine($"  JSON:  {sample.JsonPath}");

        if (!File.Exists(sample.WavPath))
        {
            Console.WriteLine("WAV file missing.");
            return;
        }

        try
        {
            var (audioMono, sampleRate) = ReadWavMono(sample.WavPath);

            // Use a fixed time window: 1024 samples at 48kHz (~21.33ms) ⇒ fewer samples at lower rates
            const int windowSamplesAt48k = 1024;
            int previewFrames = 5;

            Console.WriteLine("\nRunning unified scenarios with comparable window (~21.33ms)...");

            double nativeKhz = sampleRate / 1000.0;
            string nativeLabel = $"{nativeKhz:0.#}kHz";
            int nativeFrames = RunScenario(
                audioMono,
                sampleRate,
                targetSampleRate: sampleRate,
                windowSamplesAt48k: windowSamplesAt48k,
                previewFrames: previewFrames,
                label: nativeLabel
            );
            Console.WriteLine($"{nativeLabel} completed: {nativeFrames} frames processed");

            int resoniteFrames = RunScenario(
                audioMono,
                sampleRate,
                targetSampleRate: 48000,
                windowSamplesAt48k: windowSamplesAt48k,
                previewFrames: previewFrames,
                label: "48kHz"
            );
            Console.WriteLine($"48kHz completed: {resoniteFrames} frames processed");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
        }
    }


    private static float[] ResampleAudio(float[] input, int inputSampleRate, int outputSampleRate)
    {
        if (inputSampleRate == outputSampleRate)
            return input;

        using var resampler = new AudioResampler(inputSampleRate, outputSampleRate);
        return resampler.Resample(input);
    }

    private static int ComputeBufferSizeFrom48kWindow(int sampleRate, int windowSamplesAt48k)
    {
        return (int)Math.Round(sampleRate * (double)windowSamplesAt48k / 48000.0);
    }

    private static int RunScenario(float[] originalAudio, int originalSampleRate, int targetSampleRate, int windowSamplesAt48k, int previewFrames, string label)
    {
        int bufferSize = ComputeBufferSizeFrom48kWindow(targetSampleRate, windowSamplesAt48k);
        using var backend = new OpenLipSyncBackend();
        using var emulator = new ResoniteVisemeAnalyzerEmulator(backend, targetSampleRate, bufferSize);

        if (!emulator.IsInitialized)
        {
            Console.WriteLine($"{label}: initialization failed (check if ONNX model exists in export/ directory)");
            return 0;
        }

        var audio = targetSampleRate == originalSampleRate
            ? originalAudio
            : ResampleAudio(originalAudio, originalSampleRate, targetSampleRate);

        double frameDurationMs = 1000.0 * bufferSize / targetSampleRate;

        Console.WriteLine($"{label}: buffer={bufferSize} samples (~{frameDurationMs:0.00} ms)");

        int totalFrames = ProcessFrames(
            emulator,
            audio,
            bufferSize,
            previewFrames: previewFrames,
            previewPrinter: frame => Console.WriteLine($"[{label} t={(frame * frameDurationMs):0.000}ms] sil={emulator[0]:0.000} PP={emulator[1]:0.000} FF={emulator[2]:0.000}")
        );

        return totalFrames;
    }

    private static int ProcessFrames(ResoniteVisemeAnalyzerEmulator emulator, float[] audio, int bufferSize, int previewFrames, System.Action<int> previewPrinter)
    {
        int processedFrames = 0;
        for (int offset = 0; offset < audio.Length; offset += bufferSize)
        {
            int remaining = audio.Length - offset;
            int count = Math.Min(bufferSize, remaining);

            emulator.OnAudioUpdate(audio.AsSpan(offset, count), bufferSize);

            if (processedFrames < previewFrames)
            {
                previewPrinter(processedFrames);
            }

            processedFrames++;
        }

        return processedFrames;
    }

    // Minimal WAV reader for PCM16 and IEEE float32; returns mono float samples in [-1,1].
    private static (float[] samples, int sampleRate) ReadWavMono(string path)
    {
        using var fs = File.OpenRead(path);
        using var br = new BinaryReader(fs);

        if (new string(br.ReadChars(4)) != "RIFF") throw new InvalidDataException("Not RIFF");
        br.ReadInt32();
        if (new string(br.ReadChars(4)) != "WAVE") throw new InvalidDataException("Not WAVE");

        ushort audioFormat = 0;
        ushort numChannels = 0;
        int sampleRate = 0;
        ushort bitsPerSample = 0;
        int dataSize = 0;
        long dataPos = 0;

        while (fs.Position < fs.Length)
        {
            string id = new string(br.ReadChars(4));
            int size = br.ReadInt32();
            long next = fs.Position + size;

            if (id == "fmt ")
            {
                audioFormat = br.ReadUInt16();
                numChannels = br.ReadUInt16();
                sampleRate = br.ReadInt32();
                br.ReadInt32();
                br.ReadUInt16();
                bitsPerSample = br.ReadUInt16();
                fs.Position = next;
            }
            else if (id == "data")
            {
                dataPos = fs.Position;
                dataSize = size;
                break;
            }
            else
            {
                fs.Position = next;
            }
        }

        if (dataPos == 0) throw new InvalidDataException("Missing data chunk");
        if (numChannels == 0 || sampleRate == 0) throw new InvalidDataException("Invalid fmt chunk");

        fs.Position = dataPos;

        int bytesPerSample = bitsPerSample / 8;
        int totalSamplesPerChannel = dataSize / (bytesPerSample * numChannels);

        float[] mono = new float[totalSamplesPerChannel];

        switch (audioFormat)
        {
            case 1:
                if (bitsPerSample != 16) throw new NotSupportedException($"PCM {bitsPerSample}b not supported");
                for (int i = 0; i < totalSamplesPerChannel; i++)
                {
                    int left = br.ReadInt16();
                    int right = numChannels == 2 ? br.ReadInt16() : left;
                    mono[i] = ((left + right) * 0.5f) / 32768f;
                }
                break;

            case 3:
                if (bitsPerSample != 32) throw new NotSupportedException($"Float {bitsPerSample}b not supported");
                for (int i = 0; i < totalSamplesPerChannel; i++)
                {
                    float left = br.ReadSingle();
                    float right = numChannels == 2 ? br.ReadSingle() : left;
                    mono[i] = (left + right) * 0.5f;
                }
                break;

            default:
                throw new NotSupportedException($"Audio format {audioFormat} not supported");
        }

        return (mono, sampleRate);
    }


}