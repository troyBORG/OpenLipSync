<#
OpenLipSync MFA Alignment Script (Windows / PowerShell)
Usage:
  .\run_mfa_alignment_prepared.ps1 dev-clean
  .\run_mfa_alignment_prepared.ps1 train-clean-100
#>

[CmdletBinding()]
param(
  [Parameter(Mandatory=$true)][string]$DatasetName
)

$ErrorActionPreference = 'Stop'

# --- helpers ---
function Write-Info    { param([string]$m) Write-Host "[INFO]    $m" -ForegroundColor Cyan }
function Write-Warn    { param([string]$m) Write-Host "[WARNING] $m" -ForegroundColor Yellow }
function Write-Ok      { param([string]$m) Write-Host "[SUCCESS] $m" -ForegroundColor Green }
function Write-Err     { param([string]$m) Write-Host "[ERROR]   $m" -ForegroundColor Red }

function Assert-Command {
  param([string]$Name)
  $null = (Get-Command $Name -ErrorAction SilentlyContinue) `
          -or (Get-Command "$Name.exe" -ErrorAction SilentlyContinue)
  if (-not $?) { throw "Required command not found: $Name" }
}

# --- checks ---
try {
  Assert-Command micromamba
}
catch {
  Write-Err "micromamba is not installed or not on PATH."
  Write-Err "Install: https://mamba.readthedocs.io/en/latest/installation.html"
  exit 1
}

# Confirm 'mfa' env exists
$envList = micromamba env list 2>$null
if ($envList -notmatch '(^|\s)mfa(\s|$)') {
  Write-Err "MFA environment 'mfa' not found."
  Write-Err 'Create it: micromamba create -n mfa -c conda-forge python=3.12 montreal-forced-aligner'
  exit 1
}

# --- paths (repo-root relative) ---
$HERE        = $PSScriptRoot
$PREPARED    = Join-Path $HERE "training\data\prepared\$DatasetName"
$CACHE       = Join-Path $HERE "training\data\cache"
$TEMP_CORPUS = Join-Path $CACHE "mfa_corpus_$DatasetName"
$MFA_DIR     = Join-Path $CACHE "mfa_$DatasetName"
$OOVS_DIR    = Join-Path $MFA_DIR "oovs"
$OUT_ALIGN   = Join-Path $CACHE "out_align_$DatasetName"

Write-Info "Starting MFA alignment for dataset: $DatasetName"
Write-Info "Prepared dataset: $PREPARED"
Write-Info "Temporary corpus: $TEMP_CORPUS"
Write-Info "Output directory: $OUT_ALIGN"

if (-not (Test-Path $PREPARED)) {
  Write-Err "Prepared dataset directory not found: $PREPARED"
  Write-Err "Run your dataset preparation first."
  exit 1
}

# Count WAV/LAB
$wav = Get-ChildItem -Path $PREPARED -Filter *.wav -File -ErrorAction SilentlyContinue
$lab = Get-ChildItem -Path $PREPARED -Filter *.lab -File -ErrorAction SilentlyContinue

$wavCount = ($wav | Measure-Object).Count
$labCount = ($lab | Measure-Object).Count

if ($wavCount -eq 0 -or $labCount -eq 0) {
  Write-Err "No WAV or LAB files found in $PREPARED"
  Write-Err "WAV: $wavCount | LAB: $labCount"
  exit 1
}
if ($wavCount -ne $labCount) {
  Write-Warn "Mismatch WAV vs LAB counts (WAV: $wavCount, LAB: $labCount)"
}
Write-Info "Found $wavCount WAV and $labCount LAB"

# Create dirs
New-Item -ItemType Directory -Force $TEMP_CORPUS,$MFA_DIR,$OOVS_DIR,$OUT_ALIGN | Out-Null

# --- Build MFA corpus structure (speaker folders from file prefix) ---
Write-Info "Creating temporary corpus structure for MFA..."
Get-ChildItem -Path $PREPARED -Filter *.wav -File | ForEach-Object {
  $base = [IO.Path]::GetFileNameWithoutExtension($_.Name)
  $labf = Join-Path $PREPARED "$base.lab"
  if (Test-Path $labf) {
    $speaker = $base.Split('-')[0]
    $spkdir  = Join-Path $TEMP_CORPUS "${DatasetName}_$speaker"
    New-Item -ItemType Directory -Force $spkdir | Out-Null
    Copy-Item $_.FullName (Join-Path $spkdir "$base.wav") -Force
    Copy-Item $labf       (Join-Path $spkdir "$base.lab") -Force
  }
  else {
    Write-Warn "No matching .lab for $base.wav"
  }
}
$organized = (Get-ChildItem -Path $TEMP_CORPUS -Recurse -Filter *.wav | Measure-Object).Count
if ($organized -eq 0) {
  Write-Err "No files organized into MFA corpus structure (check naming)."
  exit 1
}
Write-Info "Organized $organized files into speaker directories."

# --- Run MFA via micromamba ---
Write-Info "Activating MFA environment and running alignment..."
# Use micromamba "run" to avoid shell activation complexity on Windows
# 1) OOV pass (non-fatal)
try {
  micromamba run -n mfa mfa find_oovs $TEMP_CORPUS english_us_arpa $OOVS_DIR | Out-Null
}
catch { Write-Warn "OOV finding failed, continuing..." }

# 2) G2P for OOVs (if any), then augment dict
$OOVS_FILE = Join-Path $OOVS_DIR "oovs_found_english_us_arpa.txt"
$OOVS_DICT = Join-Path $OOVS_DIR "oovs.dict"
if (Test-Path $OOVS_FILE -and ((Get-Item $OOVS_FILE).Length -gt 0)) {
  Write-Info "Generating pronunciations for OOVs..."
  try {
    micromamba run -n mfa mfa g2p $OOVS_FILE english_us_arpa $OOVS_DICT | Out-Null
    if (Test-Path $OOVS_DICT -and ((Get-Item $OOVS_DICT).Length -gt 0)) {
      Write-Info "Adding OOV pronunciations to ARPA dictionary..."
      micromamba run -n mfa mfa model add_words english_us_arpa $OOVS_DICT | Out-Null
    }
  }
  catch { Write-Warn "G2P/add_words failed, continuing..." }
}
else {
  Write-Info "No OOVs found; using base dictionary."
}

# 3) Align
Write-Info "Running forced alignment (this may take a while)..."
micromamba run -n mfa mfa align `
  $TEMP_CORPUS `
  english_us_arpa `
  english_us_arpa `
  $OUT_ALIGN `
  --clean `
  --output_format json

Write-Ok "Forced alignment completed."

# --- Copy JSONs back to prepared dir ---
Write-Info "Copying alignment JSONs to prepared dataset..."
$copied = 0
Get-ChildItem -Path $OUT_ALIGN -Recurse -Filter *.json |
  Where-Object { $_.Name -notlike "alignment_analysis*" } |
  ForEach-Object {
    $base = [IO.Path]::GetFileNameWithoutExtension($_.Name)
    $dst  = Join-Path $PREPARED "$base.json"
    if (Test-Path (Join-Path $PREPARED "$base.wav")) {
      Copy-Item $_.FullName $dst -Force
      $copied++
    } else {
      Write-Warn "No matching WAV for alignment: $base"
    }
  }

Write-Ok "Copied $copied alignment files."
if ($copied -lt $wavCount) {
  Write-Warn "Missing alignments: $(($wavCount - $copied))"
}

# --- Clean up temp ---
Write-Info "Cleaning up temporary files..."
Remove-Item -Recurse -Force $TEMP_CORPUS,$OUT_ALIGN,$MFA_DIR -ErrorAction SilentlyContinue

Write-Ok "MFA alignment pipeline completed for $DatasetName"
Write-Info "Prepared dataset location: $PREPARED"
