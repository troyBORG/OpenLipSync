# OpenLipSync MFA Alignment Script for Prepared Datasets (PowerShell Version)
# This script runs MFA alignment on prepared datasets with flat structure on Windows.
# Usage (from PowerShell): .\run_mfa_alignment_prepared.ps1 -DatasetName <DATASET_NAME>
# Example: .\run_mfa_alignment_prepared.ps1 -DatasetName test-clean

param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$DatasetName
)

# Helper: Write colored status messages
function Write-Status($msg)   { Write-Host "[INFO]    $msg" -ForegroundColor Cyan }
function Write-Success($msg) { Write-Host "[SUCCESS] $msg" -ForegroundColor Green }
function Write-Warning($msg) { Write-Host "[WARNING] $msg" -ForegroundColor Yellow }
function Write-ErrorMsg($msg){ Write-Host "[ERROR]   $msg" -ForegroundColor Red }

# 1. Verify micromamba installation
if (-not (Get-Command "micromamba" -ErrorAction SilentlyContinue)) {
    Write-Status "micromamba not found in PATH. Attempting to download and install it..."
    try {
        # Download and run the official micromamba install script for PowerShell
        Invoke-Expression ((Invoke-WebRequest -UseBasicParsing -Uri "https://micro.mamba.pm/install.ps1").Content)
        Write-Status "micromamba installer executed. Verifying installation..."
    } catch {
        Write-ErrorMsg "Failed to download/install micromamba. Please install micromamba manually as per documentation."
        Write-ErrorMsg "Visit: https://mamba.readthedocs.io/en/latest/installation.html for instructions."
        exit 1
    }
    # After installation, ensure the new micromamba is available in this session
    $mambaPath = "$Env:LocalAppData\micromamba\micromamba.exe"
    if (Test-Path $mambaPath) {
        # Add to PATH for the current session
        $env:PATH += [System.IO.Path]::PathSeparator + (Split-Path $mambaPath -Parent)
        Write-Success "micromamba installed successfully at $mambaPath"
    } else {
        Write-ErrorMsg "micromamba installation did not produce an expected executable."
        exit 1
    }
}

# 2. Ensure the 'mfa' Conda environment exists (create if missing)
# Check if any existing environment named "mfa" is listed
# --- Ensure 'mfa' environment is usable --------------------------------------
function Test-MfaEnv {
  try {
    # Try a harmless MFA command inside the env. Use call operator (&).
    & micromamba run -n mfa mfa version | Out-Null
    return $true
  }
  catch {
    return $false
  }
}

if (-not (Test-MfaEnv)) {
  Write-Info "Conda environment 'mfa' not found or not usable. Creating it now..."
  # IMPORTANT: create the env with micromamba *directly* (NOT via `run -n mfa`)
  & micromamba create -y -n mfa -c conda-forge python=3.12 montreal-forced-aligner
  if (-not (Test-MfaEnv)) {
    Write-Err "Failed to create Conda environment 'mfa'. Please check micromamba output."
    exit 1
  }
  Write-Ok "Environment 'mfa' is ready."
}

# 3. Define key paths (using Windows-compatible paths)
$ScriptDir      = Split-Path -Parent $MyInvocation.MyCommand.Path   # directory of this script
$PreparedDir    = Join-Path $ScriptDir "training\data\prepared\$DatasetName"
$TempCorpusDir  = Join-Path $ScriptDir "training\data\cache\mfa_corpus_$DatasetName"
$MfaWorkDir     = Join-Path $ScriptDir "training\data\cache\mfa_$DatasetName"
$OovsDir        = Join-Path $MfaWorkDir "oovs"
$TempAlignDir   = Join-Path $ScriptDir "training\data\cache\out_align_$DatasetName"

Write-Status "Starting MFA alignment for dataset: $DatasetName"
Write-Status "Prepared dataset directory:    $PreparedDir"
Write-Status "Temporary MFA corpus directory: $TempCorpusDir"
Write-Status "Alignment output directory:     $TempAlignDir"

# 4. Validate dataset directory and files
if (-not (Test-Path $PreparedDir -PathType Container)) {
    Write-ErrorMsg "Prepared dataset directory not found: $PreparedDir"
    Write-ErrorMsg "Ensure the dataset '$DatasetName' has been prepared and the path is correct."
    exit 1
}
# Count WAV and LAB files in the prepared dataset
$wavCount = (Get-ChildItem -Path $PreparedDir -Filter *.wav -File -Recurse | Measure-Object).Count
$labCount = (Get-ChildItem -Path $PreparedDir -Filter *.lab -File -Recurse | Measure-Object).Count
if ($wavCount -eq 0 -or $labCount -eq 0) {
    Write-ErrorMsg "No WAV or LAB files found in $PreparedDir"
    Write-ErrorMsg "WAV files: $wavCount, LAB files: $labCount"
    exit 1
}
if ($wavCount -ne $labCount) {
    Write-Warning "Mismatch between WAV and LAB file counts (WAV: $wavCount, LAB: $labCount)."
}
Write-Status "Found $wavCount WAV files and $labCount LAB files in prepared dataset."

# 5. Create necessary directories (corpus, MFA work, OOVs, output)
New-Item -Path $TempCorpusDir -ItemType Directory -Force | Out-Null
New-Item -Path $MfaWorkDir    -ItemType Directory -Force | Out-Null
New-Item -Path $OovsDir       -ItemType Directory -Force | Out-Null
New-Item -Path $TempAlignDir  -ItemType Directory -Force | Out-Null

# 6. Organize the corpus by speaker for MFA
Write-Status "Organizing corpus into speaker subdirectories for MFA..."
Set-Location $PreparedDir
$organizedCount = 0
# Loop through each WAV file in the prepared dataset directory (flat structure assumed)
foreach ($wavFile in Get-ChildItem -Path $PreparedDir -Filter *.wav -File) {
    $baseName = [System.IO.Path]::GetFileNameWithoutExtension($wavFile.Name)
    $labPath  = Join-Path $PreparedDir "$baseName.lab"
    if (Test-Path $labPath) {
        # Determine speaker ID (substring before first hyphen, or full name if no hyphen)
        if ($baseName.Contains('-')) {
            $speakerId = $baseName.Split('-')[0]
        } else {
            $speakerId = $baseName
        }
        $speakerDir = Join-Path $TempCorpusDir "${DatasetName}_${speakerId}"
        New-Item -Path $speakerDir -ItemType Directory -Force | Out-Null
        # Copy the WAV and LAB file into the speaker's folder
        Copy-Item -Path $wavFile.FullName -Destination $speakerDir
        Copy-Item -Path $labPath        -Destination $speakerDir
        $organizedCount++
    } else {
        Write-Warning "No corresponding .lab transcript for audio file '$($wavFile.Name)' (skipping this audio)."
    }
}
if ($organizedCount -eq 0) {
    Write-ErrorMsg "No files were organized into the MFA corpus. Alignment cannot proceed."
    exit 1
}
Write-Status "Organized $organizedCount audio files into MFA corpus directories."

# 7. Run MFA alignment pipeline within the 'mfa' environment
Write-Status "Running MFA alignment pipeline (using Conda environment 'mfa')..."

# Step 1: Find OOVs (Out-Of-Vocabulary words) in the corpus
Write-Status "Step 1: Finding OOVs in the corpus..."
& micromamba run -n mfa mfa find_oovs "$TempCorpusDir" "english_us_arpa" "$OovsDir"
if ($LASTEXITCODE -ne 0) {
    Write-Warning "OOV finding step failed (non-zero exit code), but continuing with alignment."
}
# Prepare OOV filenames for next steps
$OovListFile = Join-Path $OovsDir "oovs_found_english_us_arpa.txt"
$OovDictFile = Join-Path $OovsDir "oovs.dict"

# Step 2: Generate pronunciations for OOVs using G2P, if any OOVs were found
if (Test-Path $OovListFile -PathType Leaf -and (Get-Content $OovListFile | Select-Object -First 1)) {
    Write-Status "Step 2: Generating pronunciations for OOV words..."
    & micromamba run -n mfa mfa g2p "$OovListFile" "english_us_arpa" "$OovDictFile"
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "G2P generation step failed, but continuing with alignment."
    }
    # Step 3: Add OOV pronunciations to the ARPA dictionary model
    if (Test-Path $OovDictFile -PathType Leaf -and (Get-Content $OovDictFile | Select-Object -First 1)) {
        Write-Status "Step 3: Adding new pronunciations to ARPA dictionary model..."
        & micromamba run -n mfa mfa model add_words "english_us_arpa" "$OovDictFile"
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Adding OOV words to dictionary failed, but will proceed with alignment."
        }
    }
} else {
    Write-Status "No OOVs found in corpus. Proceeding with existing dictionary."
}

# Step 4: Perform forced alignment
Write-Status "Step 4: Running forced alignment (this may take a while)..."
& micromamba run -n mfa mfa align "$TempCorpusDir" "english_us_arpa" "english_us_arpa" "$TempAlignDir" --clean --output_format json
if ($LASTEXITCODE -ne 0) {
    Write-ErrorMsg "Forced alignment failed (MFA align returned an error)."
    exit 1
}
Write-Success "Forced alignment completed successfully."

# Step 5: Copy alignment results back to the prepared dataset
Write-Status "Step 5: Copying alignment results to prepared dataset directory..."
$alignedCount = 0
# Collect all JSON files from alignment output (exclude any alignment_analysis files)
$jsonFiles = Get-ChildItem -Path $TempAlignDir -Filter *.json -Recurse | Where-Object { $_.Name -notmatch '^alignment_analysis' }
foreach ($jsonFile in $jsonFiles) {
    $baseName = $jsonFile.BaseName  # file name without extension
    $origWav = Join-Path $PreparedDir "$baseName.wav"
    if (Test-Path $origWav) {
        Copy-Item -Path $jsonFile.FullName -Destination (Join-Path $PreparedDir "$baseName.json") -Force
        $alignedCount++
    } else {
        Write-Warning "Alignment JSON '$($jsonFile.Name)' has no matching WAV in prepared dataset (skipped)."
    }
}
Write-Success "Copied $alignedCount alignment JSON files into '$PreparedDir'."

# Print alignment statistics
Write-Status "Alignment statistics:"
Write-Host "  - Original audio files: $wavCount"
Write-Host "  - Aligned files:        $alignedCount"
if ($alignedCount -lt $wavCount) {
    $missing = $wavCount - $alignedCount
    Write-Warning "  - Missing alignments:   $missing (some audio files were not aligned!)"
}

# 8. Cleanup temporary files
Write-Status "Cleaning up temporary files and directories..."
Remove-Item -Recurse -Force $TempCorpusDir, $TempAlignDir, $MfaWorkDir

Write-Success "MFA alignment pipeline completed! Dataset '$DatasetName' is now aligned."
Write-Status  "Aligned dataset location: $PreparedDir"
