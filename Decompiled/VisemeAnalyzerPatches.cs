using System;
using System.IO;
using System.Runtime.CompilerServices;
using HarmonyLib;
using Elements.Core; // UniLog
using FrooxEngine;
using OpenLipSyncCompat;

namespace VisemesAtHome.Patches;

[HarmonyPatch]
internal static class VisemeAnalyzerPatches
{
    private static readonly string[] OvrNames =
    {
        "sil","PP","FF","TH","DD","kk","CH","SS","nn","RR","aa","E","ih","oh","ou"
    };

    // Per-instance state (stores our compat OVR context)
    private static readonly ConditionalWeakTable<VisemeAnalyzer, AnalyzerState> State = new();

    private sealed class AnalyzerState : IDisposable
    {
        public uint ContextId;
        public OpenLipSync.Frame Frame = new();
        public float[] LastWeights = new float[15];
        public float LastSmooth = -1f;
        public int[] Remap = IdentityRemap;
        public bool RemapParsed;
        public void Dispose()
        {
            try { if (ContextId != 0) OpenLipSync.DestroyContext(ContextId); } catch { /* ignore */ }
            ContextId = 0;
        }
    }

    // Private fields on FrooxEngine.VisemeAnalyzer
    private static readonly AccessTools.FieldRef<VisemeAnalyzer, object?> f_analysisContext = AccessTools.FieldRefAccess<VisemeAnalyzer, object?>("analysisContext");
    private static readonly AccessTools.FieldRef<VisemeAnalyzer, float[]?> f_buffer          = AccessTools.FieldRefAccess<VisemeAnalyzer, float[]>("buffer");
    private static readonly AccessTools.FieldRef<VisemeAnalyzer, float[]>  f_analysis        = AccessTools.FieldRefAccess<VisemeAnalyzer, float[]>("analysis");
    private static readonly AccessTools.FieldRef<VisemeAnalyzer, bool>     f_hasRemoteSource = AccessTools.FieldRefAccess<VisemeAnalyzer, bool>("_hasRemoteSource");
    private static readonly AccessTools.FieldRef<VisemeAnalyzer, Sync<float>> f_smoothing     = AccessTools.FieldRefAccess<VisemeAnalyzer, Sync<float>>("Smoothing");

    // After Awake, ensure OVR path is disabled so it won't race our results
    [HarmonyPostfix]
    [HarmonyPatch(typeof(VisemeAnalyzer), "OnAwake")]
    private static void OnAwake_Postfix(VisemeAnalyzer __instance)
    {
        try
        {
            var ctx = f_analysisContext(__instance);
            if (ctx != null)
            {
                // Best-effort dispose to avoid leaks
                try { (ctx as IDisposable)?.Dispose(); } catch { /* ignore */ }
                f_analysisContext(__instance) = null;
                VahLog.Info("Disabled OVRLipSyncContext for VisemeAnalyzer instance.");
            }
        }
        catch (Exception ex)
        {
            VahLog.Warn("Failed to disable OVRLipSyncContext: " + ex.Message);
        }

        // Ensure we have a state bucket
        var st = State.GetValue(__instance, _ => new AnalyzerState());
        if (!st.RemapParsed)
        {
            st.Remap = ParseRemapEnv();
            st.RemapParsed = true;
            if (st.Remap != IdentityRemap)
                VahLog.Warn("Using custom viseme remap from VAH_REMAP.");
        }
    }

    [HarmonyPostfix]
    [HarmonyPatch(typeof(VisemeAnalyzer), "OnDispose")]
    private static void OnDispose_Postfix(VisemeAnalyzer __instance)
    {
        if (State.TryGetValue(__instance, out var st))
        {
            st.Dispose();
        }
    }

    // After audio update, run ONNX and write 15 viseme weights; keep laughter at index 15 as 0 for now
    [HarmonyPostfix]
    [HarmonyPatch(typeof(VisemeAnalyzer), "OnAudioUpdate")]
    private static void OnAudioUpdate_Postfix(VisemeAnalyzer __instance)
    {
        try
        {
            // Respect remote source: do not compute when driven by someone else
            if (f_hasRemoteSource(__instance))
                return;

            var buf = f_buffer(__instance);
            var ana = f_analysis(__instance);
            if (buf == null || buf.Length == 0 || ana == null || ana.Length < 16)
                return;

            var st = State.GetValue(__instance, _ => new AnalyzerState());

            // Determine target model IO: resample engine buffer (~48kHz, e.g., 1024 samples) to 16kHz 20ms (320 samples)
            const int targetSr = 16000;
            const int targetFrameMs = 20;
            int targetFrame = (int)MathF.Round(targetSr * (targetFrameMs / 1000f)); // 320

            // Read smoothing from component so user control applies; clamp to sane range
            float smooth = 0.65f;
            try { smooth = Math.Clamp(f_smoothing(__instance).Value, 0f, 0.98f); } catch { }

            // (Re)create context if missing or config changed
            if (st.ContextId == 0 || Math.Abs(st.LastSmooth - smooth) > 0.01f)
            {
                if (st.ContextId != 0)
                {
                    OpenLipSync.DestroyContext(st.ContextId);
                    st.ContextId = 0;
                }

                // Pick model path relative to the mod's DLL
                string baseDir = AppContext.BaseDirectory;
                string modelPath = Path.Combine(baseDir, "Models", "bf16.onnx");
                if (!File.Exists(modelPath))
                {
                    // fallback to fp32 if bf16 is missing
                    string fp32 = Path.Combine(baseDir, "Models", "fp32.onnx");
                    if (File.Exists(fp32)) modelPath = fp32;
                }

                VahLog.Info($"Creating OpenLipSyncCompat context: model='{modelPath}', sr={targetSr} Hz, frameMs={targetFrameMs}, smooth={smooth:0.00}");
                var res = OpenLipSync.CreateContext(out var id, OpenLipSync.ContextProviders.Enhanced, targetSr, targetFrameMs, modelPath, smooth);
                if (res != OpenLipSync.Result.Success)
                {
                    VahLog.Error("Failed to create OpenLipSyncCompat context: " + res);
                    return;
                }
                st.ContextId = id;
                st.LastSmooth = smooth;
            }

            // Silence gate (disabled for debugging classification)
            float rms = 1f;
            const float silThr = 0f;
            const float silDecay = 0.85f;

            ReadOnlySpan<float> w;
            if (rms >= silThr)
            {
                var pr = OpenLipSync.ProcessFrame(st.ContextId, buf, buf.Length, st.Frame);
                if (pr != OpenLipSync.Result.Success)
                {
                    VahLog.Error("OpenLipSyncCompat.ProcessFrame failed: " + pr);
                    return;
                }
                for (int i = 0; i < 15; i++) st.LastWeights[i] = st.Frame.visemes[i];
                w = st.LastWeights;
            }
            else
            {
                // Decay previous weights toward silence
                for (int i = 1; i < 15; i++) st.LastWeights[i] *= silDecay;
                float sumOthers = 0f; for (int i = 1; i < 15; i++) sumOthers += st.LastWeights[i];
                st.LastWeights[0] = 1f - Math.Clamp(sumOthers, 0f, 0.999f);
                w = st.LastWeights;
            }
            // Debug: show raw model top before any remap
            if (VahLog.DebugEnabled)
            {
                Span<int> ridx = stackalloc int[15];
                for (int i = 0; i < 15; i++) ridx[i] = i;
                for (int i = 0; i < 4; i++)
                {
                    int maxJ = i;
                    for (int j = i + 1; j < 15; j++) if (w[ridx[j]] > w[ridx[maxJ]]) maxJ = j;
                    (ridx[i], ridx[maxJ]) = (ridx[maxJ], ridx[i]);
                }
                VahLog.Debug($"RawTop: {OvrNames[ridx[0]]}:{w[ridx[0]]:0.00} {OvrNames[ridx[1]]}:{w[ridx[1]]:0.00} {OvrNames[ridx[2]]}:{w[ridx[2]]:0.00} {OvrNames[ridx[3]]}:{w[ridx[3]]:0.00} | SS={w[7]:0.00} aa={w[10]:0.00}");
            }

            // Apply optional remap (identity by default)
            for (int i = 0; i < 15; i++) ana[st.Remap[i]] = w[i];
            ana[15] = 0f; // Laughter placeholder

            if (VahLog.DebugEnabled)
            {
                // Log top-4 with names
                Span<int> idx = stackalloc int[15];
                for (int i = 0; i < 15; i++) idx[i] = i;
                for (int i = 0; i < 4; i++)
                {
                    int maxJ = i;
                    for (int j = i + 1; j < 15; j++) if (ana[idx[j]] > ana[idx[maxJ]]) maxJ = j;
                    (idx[i], idx[maxJ]) = (idx[maxJ], idx[i]);
                }
                VahLog.Debug($"Top: {OvrNames[idx[0]]}:{ana[idx[0]]:0.00} {OvrNames[idx[1]]}:{ana[idx[1]]:0.00} {OvrNames[idx[2]]}:{ana[idx[2]]:0.00} {OvrNames[idx[3]]}:{ana[idx[3]]:0.00} | SS={ana[7]:0.00}");
            }
        }
        catch (Exception ex)
        {
            VahLog.Error("OpenLipSyncCompat analysis failed: " + ex.Message);
        }
    }

    // Resampling helpers removed; handled by OpenLipSyncCompat

    private static readonly int[] IdentityRemap = new int[] { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14 };
    private static int[] ParseRemapEnv()
    {
        try
        {
            var env = Environment.GetEnvironmentVariable("VAH_REMAP");
            if (string.IsNullOrWhiteSpace(env)) return IdentityRemap;
            var parts = env.Split(',');
            if (parts.Length != 15) return IdentityRemap;
            var map = new int[15];
            for (int i = 0; i < 15; i++)
            {
                if (!int.TryParse(parts[i].Trim(), out int v)) return IdentityRemap;
                if (v < 0 || v >= 15) return IdentityRemap;
                map[i] = v;
            }
            return map;
        }
        catch
        {
            return IdentityRemap;
        }
    }
}


