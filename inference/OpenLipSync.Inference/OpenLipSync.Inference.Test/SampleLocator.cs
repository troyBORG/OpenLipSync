using System.Diagnostics.CodeAnalysis;

namespace OpenLipSync.Inference.Test;

public static class SampleLocator
{
    public sealed record SamplePaths(string WavPath, string LabPath, string JsonPath);

    public static SamplePaths? FindAnySample()
    {
        var repoRoot = FindRepoRoot();
        if (repoRoot is null) return null;

        var devCleanDir = Path.Combine(repoRoot, "training", "data", "prepared", "dev-clean");
        if (!Directory.Exists(devCleanDir)) return null;

        // Map base name -> discovered extensions
        var baseNameToPaths = new Dictionary<string, Dictionary<string, string>>(StringComparer.OrdinalIgnoreCase);

        foreach (var filePath in Directory.EnumerateFiles(devCleanDir, "*.*", SearchOption.AllDirectories))
        {
            var ext = Path.GetExtension(filePath);
            if (!IsRelevantExtension(ext)) continue;

            var baseName = Path.GetFileNameWithoutExtension(filePath);
            if (!baseNameToPaths.TryGetValue(baseName, out var byExt))
            {
                byExt = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
                baseNameToPaths[baseName] = byExt;
            }
            byExt[ext] = filePath;
        }

        foreach (var kvp in baseNameToPaths)
        {
            var byExt = kvp.Value;
            if (byExt.TryGetValue(".wav", out var wav)
                && byExt.TryGetValue(".lab", out var lab)
                && byExt.TryGetValue(".json", out var json))
            {
                return new SamplePaths(wav, lab, json);
            }
        }

        return null;
    }

    private static bool IsRelevantExtension(string? ext)
    {
        if (string.IsNullOrEmpty(ext)) return false;
        return ext.Equals(".wav", StringComparison.OrdinalIgnoreCase)
            || ext.Equals(".lab", StringComparison.OrdinalIgnoreCase)
            || ext.Equals(".json", StringComparison.OrdinalIgnoreCase);
    }

    private static string? FindRepoRoot()
    {
        // Walk up from current executable directory until we see the top-level markers
        var current = AppContext.BaseDirectory;

        while (!string.IsNullOrEmpty(current))
        {
            if (LooksLikeRepoRoot(current)) return current;
            var parent = Directory.GetParent(current);
            if (parent is null) break;
            current = parent.FullName;
        }

        // Fallback: try current working directory when running from IDE
        current = Directory.GetCurrentDirectory();
        while (!string.IsNullOrEmpty(current))
        {
            if (LooksLikeRepoRoot(current)) return current;
            var parent = Directory.GetParent(current);
            if (parent is null) break;
            current = parent.FullName;
        }

        return null;
    }

    private static bool LooksLikeRepoRoot(string dir)
    {
        // Heuristics: presence of these files/directories at root
        var pyproject = Path.Combine(dir, "pyproject.toml");
        var trainingDir = Path.Combine(dir, "training");
        return File.Exists(pyproject) && Directory.Exists(trainingDir);
    }
}


