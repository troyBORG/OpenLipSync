// FrooxEngine.VisemeAnalyzer
using System;
using Elements.Assets;
using Elements.Core;
using FrooxEngine;

[Category(new string[] { "Media/Utility" })]
public class VisemeAnalyzer : Component
{
	public readonly SyncRef<IWorldAudioDataSource> Source;

	public readonly SyncRef<MultiValueStream<float>> RemoteSource;

	[Range(0f, 1f, "0.00")]
	public readonly Sync<float> Smoothing;

	[Range(0f, 1f, "0.00")]
	public readonly RawOutput<float> Silence;

	[Range(0f, 1f, "0.00")]
	public readonly RawOutput<float> PP;

	[Range(0f, 1f, "0.00")]
	public readonly RawOutput<float> FF;

	[Range(0f, 1f, "0.00")]
	public readonly RawOutput<float> TH;

	[Range(0f, 1f, "0.00")]
	public readonly RawOutput<float> DD;

	[Range(0f, 1f, "0.00")]
	public readonly RawOutput<float> kk;

	[Range(0f, 1f, "0.00")]
	public readonly RawOutput<float> CH;

	[Range(0f, 1f, "0.00")]
	public readonly RawOutput<float> SS;

	[Range(0f, 1f, "0.00")]
	public readonly RawOutput<float> nn;

	[Range(0f, 1f, "0.00")]
	public readonly RawOutput<float> RR;

	[Range(0f, 1f, "0.00")]
	public readonly RawOutput<float> aa;

	[Range(0f, 1f, "0.00")]
	public readonly RawOutput<float> E;

	[Range(0f, 1f, "0.00")]
	public readonly RawOutput<float> ih;

	[Range(0f, 1f, "0.00")]
	public readonly RawOutput<float> oh;

	[Range(0f, 1f, "0.00")]
	public readonly RawOutput<float> ou;

	[Range(0f, 1f, "0.00")]
	public readonly RawOutput<float> LaughterProbability;

	private OVRLipSyncContext analysisContext;

	private volatile bool _analysisRunning;

	private float[] buffer;

	private float[] analysis;

	private IAudioStream _currentAudioStream;

	private bool _hasRemoteSource;

	private Action _onAnalyzed;

	public RawOutput<float> this[Viseme viseme] => viseme switch
	{
		Viseme.Silence => Silence, 
		Viseme.PP => PP, 
		Viseme.FF => FF, 
		Viseme.TH => TH, 
		Viseme.DD => DD, 
		Viseme.kk => kk, 
		Viseme.CH => CH, 
		Viseme.SS => SS, 
		Viseme.nn => nn, 
		Viseme.RR => RR, 
		Viseme.aa => aa, 
		Viseme.E => E, 
		Viseme.ih => ih, 
		Viseme.oh => oh, 
		Viseme.ou => ou, 
		Viseme.Laughter => LaughterProbability, 
		_ => null, 
	};

	protected override void OnAwake()
	{
		base.OnAwake();
		analysis = new float[16];
		Smoothing.Value = 0.7f;
		_onAnalyzed = OnAnalyzed;
		if (base.AudioSystem.OVRLipSync.IsInitialized)
		{
			analysisContext = new OVRLipSyncContext(base.AudioSystem.OVRLipSync);
		}
	}

	protected override void OnDispose()
	{
		analysisContext?.Dispose();
		analysisContext = null;
		_currentAudioStream = null;
		base.OnDispose();
	}

	protected override void OnChanges()
	{
		base.OnChanges();
		IAudioStream audioStream = Source.Target as IAudioStream;
		if (_currentAudioStream != audioStream)
		{
			if (audioStream == null)
			{
				RemoteSource.Target = null;
			}
			else if (audioStream.IsOwnedByLocalUser)
			{
				if (analysisContext == null)
				{
					RemoteSource.Target = null;
				}
				else
				{
					RemoteSource.Target = base.LocalUser.GetStreamOrAdd($"Visemes.{audioStream.ReferenceID}", delegate(MultiValueStream<float> s)
					{
						s.SetUpdatePeriod(0u, 0u);
						s.Encoding = ValueEncoding.Quantized;
						s.FullFrameBits = 6;
						s.FullFrameMin = 0f;
						s.FullFrameMax = 1f;
						s.Count = 16;
					});
				}
			}
			_currentAudioStream = audioStream;
		}
		_hasRemoteSource = !(RemoteSource.Target?.IsOwnedByLocalUser ?? true);
	}

	protected override void OnCommonUpdate()
	{
		MultiValueStream<float> streamSource = RemoteSource.Target;
		if (streamSource != null)
		{
			if (!streamSource.IsOwnedByLocalUser)
			{
				if (streamSource.HasValidData)
				{
					for (int j = 0; j < 16; j++)
					{
						this[(Viseme)j].Value = streamSource[j];
					}
				}
				return;
			}
			for (int k = 0; k < 16; k++)
			{
				streamSource[k] = analysis[k];
			}
		}
		for (int i = 0; i < analysis.Length; i++)
		{
			this[(Viseme)i].Value = analysis[i];
		}
	}

	protected override void OnAudioUpdate()
	{
		if (!base.Enabled)
		{
			Array.Clear(analysis, 0, analysis.Length);
		}
		else
		{
			if (_hasRemoteSource || _analysisRunning)
			{
				return;
			}
			_analysisRunning = true;
			IWorldAudioDataSource _target = Source.Target;
			if (_target != null)
			{
				int readSamples = base.Engine.AudioSystem.SimulationFrameSize;
				buffer = buffer.EnsureExactSize(readSamples);
				_target.Read(buffer.AsMonoBuffer(), base.Audio.Simulator);
				if (analysisContext != null)
				{
					analysisContext.Analyze(buffer, analysis, _onAnalyzed);
				}
				else
				{
					OnAnalyzed();
				}
			}
			else
			{
				Array.Clear(analysis, 0, analysis.Length);
				_analysisRunning = false;
			}
		}
	}

	private void OnAnalyzed()
	{
		MultiValueStream<float> streamSource = RemoteSource.Target;
		if (streamSource != null && streamSource.IsOwnedByLocalUser)
		{
			streamSource.ForceUpdate();
		}
		_analysisRunning = false;
	}

	protected override void InitializeSyncMembers()
	{
		base.InitializeSyncMembers();
		Source = new SyncRef<IWorldAudioDataSource>();
		RemoteSource = new SyncRef<MultiValueStream<float>>();
		Smoothing = new Sync<float>();
		Silence = new RawOutput<float>();
		PP = new RawOutput<float>();
		FF = new RawOutput<float>();
		TH = new RawOutput<float>();
		DD = new RawOutput<float>();
		kk = new RawOutput<float>();
		CH = new RawOutput<float>();
		SS = new RawOutput<float>();
		nn = new RawOutput<float>();
		RR = new RawOutput<float>();
		aa = new RawOutput<float>();
		E = new RawOutput<float>();
		ih = new RawOutput<float>();
		oh = new RawOutput<float>();
		ou = new RawOutput<float>();
		LaughterProbability = new RawOutput<float>();
	}

	public override ISyncMember GetSyncMember(int index)
	{
		return index switch
		{
			0 => persistent, 
			1 => updateOrder, 
			2 => EnabledField, 
			3 => Source, 
			4 => RemoteSource, 
			5 => Smoothing, 
			6 => Silence, 
			7 => PP, 
			8 => FF, 
			9 => TH, 
			10 => DD, 
			11 => kk, 
			12 => CH, 
			13 => SS, 
			14 => nn, 
			15 => RR, 
			16 => aa, 
			17 => E, 
			18 => ih, 
			19 => oh, 
			20 => ou, 
			21 => LaughterProbability, 
			_ => throw new ArgumentOutOfRangeException(), 
		};
	}

	public static VisemeAnalyzer __New()
	{
		return new VisemeAnalyzer();
	}
}

// FrooxEngine.Viseme
using Elements.Data;

[DataModelType]
public enum Viseme
{
	Silence,
	PP,
	FF,
	TH,
	DD,
	kk,
	CH,
	SS,
	nn,
	RR,
	aa,
	E,
	ih,
	oh,
	ou,
	Laughter,
	COUNT
}

// FrooxEngine.OVRLipSyncContext
using System;
using System.Threading.Tasks.Dataflow;
using Elements.Core;
using FrooxEngine;

public class OVRLipSyncContext : IDisposable
{
	private readonly struct AnalysisTask
	{
		public readonly OVRLipSyncContext context;

		public readonly float[] buffer;

		public readonly float[] analysis;

		public readonly Action onDone;

		public AnalysisTask(OVRLipSyncContext context, float[] buffer, float[] analysis, Action onDone)
		{
			this.context = context;
			this.buffer = buffer;
			this.analysis = analysis;
			this.onDone = onDone;
		}
	}

	private static ActionBlock<AnalysisTask> processor = new ActionBlock<AnalysisTask>(delegate(AnalysisTask task)
	{
		task.context.RunAnalysis(in task);
	}, new ExecutionDataflowBlockOptions
	{
		MaxDegreeOfParallelism = 1,
		EnsureOrdered = false
	});

	private OVRLipSyncInterface ovrLipSync;

	private uint context;

	private OVRLipSyncInterface.Frame frame = new OVRLipSyncInterface.Frame();

	public bool IsInitialized => context != 0;

	public float Smoothing { get; private set; } = -1f;


	public OVRLipSyncContext(OVRLipSyncInterface ovrLipSync)
	{
		try
		{
			this.ovrLipSync = ovrLipSync;
			OVRLipSyncInterface.Result result = ovrLipSync.CreateContext(ref context, OVRLipSyncInterface.ContextProviders.Enhanced_with_Laughter, ovrLipSync.SampleRate, enableAcceleration: true);
			if (result != 0)
			{
				context = 0u;
				UniLog.Error("Error initializing OVRLipSyncInterface: " + result);
			}
		}
		catch (Exception ex)
		{
			context = 0u;
			UniLog.Error("Exception initializing OVRLipSyncInterface: " + ex);
		}
	}

	public void Update(float smoothing)
	{
		if (smoothing == Smoothing)
		{
			return;
		}
		Smoothing = smoothing;
		lock (this)
		{
			if (context != 0)
			{
				int clampedSmoothing = MathX.Clamp((int)(Smoothing * 100f), 0, 100);
				OVRLipSyncInterface.Result result = ovrLipSync.SendSignal(context, OVRLipSyncInterface.Signals.VisemeSmoothing, clampedSmoothing, 0);
				if (result != 0)
				{
					UniLog.Error("Error setting OVRLipSyncInterface smoothing: " + result, stackTrace: false);
				}
			}
		}
	}

	public void Dispose()
	{
		lock (this)
		{
			OVRLipSyncInterface.Result result = ovrLipSync.DestroyContext(context);
			context = 0u;
			if (result != 0)
			{
				UniLog.Error("Error destroying OVRLipSyncInterface: " + result);
			}
		}
	}

	public void Analyze(float[] audioData, float[] analysis, Action onDone)
	{
		processor.Post(new AnalysisTask(this, audioData, analysis, onDone));
	}

	private void RunAnalysis(in AnalysisTask task)
	{
		try
		{
			lock (this)
			{
				if (context == 0)
				{
					return;
				}
				OVRLipSyncInterface.Result result = ovrLipSync.ProcessFrame(context, task.buffer, frame, stereo: false);
				if (result != 0)
				{
					UniLog.Error("Error processing frame with OVRLipSyncInterface: " + result);
					return;
				}
				for (int i = 0; i < task.analysis.Length; i++)
				{
					task.analysis[i] = GetData((Viseme)i);
				}
			}
		}
		finally
		{
			task.onDone();
		}
	}

	private float GetData(Viseme v)
	{
		if (v == Viseme.Laughter)
		{
			return frame.laughterScore;
		}
		return frame.Visemes[(int)v];
	}
}

// FrooxEngine.OVRLipSyncInterface
using System;
using System.Runtime.InteropServices;
using Elements.Core;
using FrooxEngine;

public class OVRLipSyncInterface : IDisposable
{
	public enum Result
	{
		Success = 0,
		Unknown = -2200,
		CannotCreateContext = -2201,
		InvalidParam = -2202,
		BadSampleRate = -2203,
		MissingDLL = -2204,
		BadVersion = -2205,
		UndefinedFunction = -2206
	}

	public enum AudioDataType
	{
		S16_Mono,
		S16_Stereo,
		F32_Mono,
		F32_Stereo
	}

	public enum Viseme
	{
		sil,
		PP,
		FF,
		TH,
		DD,
		kk,
		CH,
		SS,
		nn,
		RR,
		aa,
		E,
		ih,
		oh,
		ou
	}

	public enum Signals
	{
		VisemeOn,
		VisemeOff,
		VisemeAmount,
		VisemeSmoothing,
		LaughterAmount
	}

	public enum ContextProviders
	{
		Original,
		Enhanced,
		Enhanced_with_Laughter
	}

	/// NOTE: Opaque typedef for lip-sync context is an unsigned int (uint)
	/// Current phoneme frame results
	[Serializable]
	public class Frame
	{
		public int frameNumber;

		public int frameDelay;

		public float[] Visemes = new float[VisemeCount];

		public float laughterScore;

		public void CopyInput(Frame input)
		{
			frameNumber = input.frameNumber;
			frameDelay = input.frameDelay;
			input.Visemes.CopyTo(Visemes, 0);
			laughterScore = input.laughterScore;
		}

		public void Reset()
		{
			frameNumber = 0;
			frameDelay = 0;
			Array.Clear(Visemes, 0, VisemeCount);
			laughterScore = 0f;
		}
	}

	public static readonly int VisemeCount = Enum.GetNames(typeof(Viseme)).Length;

	public static readonly int SignalCount = Enum.GetNames(typeof(Signals)).Length;

	public const string strOVRLS = "OVRLipSync";

	private Result initResult = Result.Unknown;

	public bool IsInitialized => initResult == Result.Success;

	public int SampleRate { get; private set; }

	public int BufferSize { get; private set; }

	[DllImport("OVRLipSync")]
	private static extern int ovrLipSyncDll_Initialize(int samplerate, int buffersize);

	[DllImport("OVRLipSync")]
	private static extern void ovrLipSyncDll_Shutdown();

	[DllImport("OVRLipSync")]
	private static extern IntPtr ovrLipSyncDll_GetVersion(ref int Major, ref int Minor, ref int Patch);

	[DllImport("OVRLipSync")]
	private static extern int ovrLipSyncDll_CreateContextEx(ref uint context, ContextProviders provider, int sampleRate, bool enableAcceleration);

	[DllImport("OVRLipSync")]
	private static extern int ovrLipSyncDll_CreateContextWithModelFile(ref uint context, ContextProviders provider, string modelPath, int sampleRate, bool enableAcceleration);

	[DllImport("OVRLipSync")]
	private static extern int ovrLipSyncDll_DestroyContext(uint context);

	[DllImport("OVRLipSync")]
	private static extern int ovrLipSyncDll_ResetContext(uint context);

	[DllImport("OVRLipSync")]
	private static extern int ovrLipSyncDll_SendSignal(uint context, Signals signal, int arg1, int arg2);

	[DllImport("OVRLipSync")]
	private static extern int ovrLipSyncDll_ProcessFrameEx(uint context, IntPtr audioBuffer, uint bufferSize, AudioDataType dataType, ref int frameNumber, ref int frameDelay, float[] visemes, int visemeCount, ref float laughterScore, float[] laughterCategories, int laughterCategoriesLength);

	public OVRLipSyncInterface(int sampleRate, int frameSize)
	{
		initResult = Initialize(sampleRate, frameSize);
		if (initResult != 0)
		{
			UniLog.Warning("Failed to initialize OVRLipSync: " + initResult);
		}
	}

	public void Dispose()
	{
		ovrLipSyncDll_Shutdown();
		initResult = Result.Unknown;
	}

	private Result Initialize(int sampleRate, int frameSize)
	{
		SampleRate = sampleRate;
		BufferSize = frameSize;
		UniLog.Log($"Initializing OVRLipSync. SampleRate: {SampleRate}, BufferSize: {BufferSize}");
		try
		{
			initResult = (Result)ovrLipSyncDll_Initialize(SampleRate, BufferSize);
		}
		catch (DllNotFoundException)
		{
			initResult = Result.MissingDLL;
		}
		return initResult;
	}

	/// <summary>
	/// Creates a lip-sync context.
	/// </summary>
	/// <returns>error code</returns>
	/// <param name="context">Context.</param>
	/// <param name="provider">Provider.</param>
	/// <param name="enableAcceleration">Enable DSP Acceleration.</param>
	public Result CreateContext(ref uint context, ContextProviders provider, int sampleRate = 0, bool enableAcceleration = false)
	{
		if (!IsInitialized)
		{
			return Result.CannotCreateContext;
		}
		return (Result)ovrLipSyncDll_CreateContextEx(ref context, provider, sampleRate, enableAcceleration);
	}

	/// <summary>
	/// Creates a lip-sync context with specified model file.
	/// </summary>
	/// <returns>error code</returns>
	/// <param name="context">Context.</param>
	/// <param name="provider">Provider.</param>
	/// <param name="modelPath">Model Dir.</param>
	/// <param name="sampleRate">Sampling Rate.</param>
	/// <param name="enableAcceleration">Enable DSP Acceleration.</param>
	public Result CreateContextWithModelFile(ref uint context, ContextProviders provider, string modelPath, int sampleRate = 0, bool enableAcceleration = false)
	{
		if (!IsInitialized)
		{
			return Result.CannotCreateContext;
		}
		return (Result)ovrLipSyncDll_CreateContextWithModelFile(ref context, provider, modelPath, sampleRate, enableAcceleration);
	}

	/// <summary>
	/// Destroy a lip-sync context.
	/// </summary>
	/// <returns>The context.</returns>
	/// <param name="context">Context.</param>
	public Result DestroyContext(uint context)
	{
		if (!IsInitialized)
		{
			return Result.Unknown;
		}
		return (Result)ovrLipSyncDll_DestroyContext(context);
	}

	/// <summary>
	/// Resets the context.
	/// </summary>
	/// <returns>error code</returns>
	/// <param name="context">Context.</param>
	public Result ResetContext(uint context)
	{
		if (!IsInitialized)
		{
			return Result.Unknown;
		}
		return (Result)ovrLipSyncDll_ResetContext(context);
	}

	/// <summary>
	/// Sends a signal to the lip-sync engine.
	/// </summary>
	/// <returns>error code</returns>
	/// <param name="context">Context.</param>
	/// <param name="signal">Signal.</param>
	/// <param name="arg1">Arg1.</param>
	/// <param name="arg2">Arg2.</param>
	public Result SendSignal(uint context, Signals signal, int arg1, int arg2)
	{
		if (!IsInitialized)
		{
			return Result.Unknown;
		}
		return (Result)ovrLipSyncDll_SendSignal(context, signal, arg1, arg2);
	}

	/// <summary>
	///  Process float[] audio buffer by lip-sync engine.
	/// </summary>
	/// <returns>error code</returns>
	/// <param name="context">Context.</param>
	/// <param name="audioBuffer"> PCM audio buffer.</param>
	/// <param name="frame">Lip-sync Frame.</param>
	/// <param name="stereo">Whether buffer is part of stereo or mono stream.</param>
	public Result ProcessFrame(uint context, float[] audioBuffer, Frame frame, bool stereo = true)
	{
		if (!IsInitialized)
		{
			return Result.Unknown;
		}
		AudioDataType dataType = (stereo ? AudioDataType.F32_Stereo : AudioDataType.F32_Mono);
		uint numSamples = (uint)(stereo ? (audioBuffer.Length / 2) : audioBuffer.Length);
		GCHandle handle = GCHandle.Alloc(audioBuffer, GCHandleType.Pinned);
		int result = ovrLipSyncDll_ProcessFrameEx(context, handle.AddrOfPinnedObject(), numSamples, dataType, ref frame.frameNumber, ref frame.frameDelay, frame.Visemes, frame.Visemes.Length, ref frame.laughterScore, null, 0);
		handle.Free();
		return (Result)result;
	}

	/// <summary>
	///  Process short[] audio buffer by lip-sync engine.
	/// </summary>
	/// <returns>error code</returns>
	/// <param name="context">Context.</param>
	/// <param name="audioBuffer"> PCM audio buffer.</param>
	/// <param name="frame">Lip-sync Frame.</param>
	/// <param name="stereo">Whether buffer is part of stereo or mono stream.</param>
	public Result ProcessFrame(uint context, short[] audioBuffer, Frame frame, bool stereo = true)
	{
		if (!IsInitialized)
		{
			return Result.Unknown;
		}
		AudioDataType dataType = (stereo ? AudioDataType.S16_Stereo : AudioDataType.S16_Mono);
		uint numSamples = (uint)(stereo ? (audioBuffer.Length / 2) : audioBuffer.Length);
		GCHandle handle = GCHandle.Alloc(audioBuffer, GCHandleType.Pinned);
		int result = ovrLipSyncDll_ProcessFrameEx(context, handle.AddrOfPinnedObject(), numSamples, dataType, ref frame.frameNumber, ref frame.frameDelay, frame.Visemes, frame.Visemes.Length, ref frame.laughterScore, null, 0);
		handle.Free();
		return (Result)result;
	}
}
