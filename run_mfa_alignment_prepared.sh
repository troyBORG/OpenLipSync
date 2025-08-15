#!/bin/bash

# OpenLipSync MFA Alignment Script for Prepared Datasets
# This script runs MFA alignment on prepared datasets with flat structure
# 
# Usage: ./run_mfa_alignment_prepared.sh DATASET_NAME
# Example: ./run_mfa_alignment_prepared.sh test-clean

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check arguments
if [ $# -ne 1 ]; then
    print_error "Usage: $0 DATASET_NAME"
    print_error "Example: $0 test-clean"
    exit 1
fi

DATASET_NAME="$1"

# Check if micromamba is available
if ! command_exists micromamba; then
    print_error "micromamba is not installed or not in PATH"
    print_error "Please install micromamba first: https://mamba.readthedocs.io/en/latest/installation.html"
    exit 1
fi

# Check if MFA environment exists
if ! micromamba env list | grep -q "mfa"; then
    print_error "MFA environment not found"
    print_error "Please create it first:"
    print_error "micromamba create -n mfa -c conda-forge python=3.12 montreal-forced-aligner"
    exit 1
fi

# Define paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREPARED_DIR="${SCRIPT_DIR}/training/data/prepared/${DATASET_NAME}"
TEMP_CORPUS="${SCRIPT_DIR}/training/data/cache/mfa_corpus_${DATASET_NAME}"
MFA_DIR="${SCRIPT_DIR}/training/data/cache/mfa_${DATASET_NAME}"
OOVS_DIR="${MFA_DIR}/oovs"
TEMP_OUT_ALIGN="${SCRIPT_DIR}/training/data/cache/out_align_${DATASET_NAME}"

print_status "Starting MFA alignment for dataset: ${DATASET_NAME}"
print_status "Prepared dataset: ${PREPARED_DIR}"
print_status "Temporary corpus: ${TEMP_CORPUS}"
print_status "Output directory: ${TEMP_OUT_ALIGN}"

# Check if prepared dataset exists
if [ ! -d "${PREPARED_DIR}" ]; then
    print_error "Prepared dataset directory not found: ${PREPARED_DIR}"
    print_error "Please ensure the dataset has been prepared first."
    exit 1
fi

# Check if we have WAV and LAB files
WAV_COUNT=$(find "${PREPARED_DIR}" -name "*.wav" | wc -l)
LAB_COUNT=$(find "${PREPARED_DIR}" -name "*.lab" | wc -l)

if [ "${WAV_COUNT}" -eq 0 ] || [ "${LAB_COUNT}" -eq 0 ]; then
    print_error "No WAV or LAB files found in ${PREPARED_DIR}"
    print_error "WAV files: ${WAV_COUNT}, LAB files: ${LAB_COUNT}"
    exit 1
fi

if [ "${WAV_COUNT}" -ne "${LAB_COUNT}" ]; then
    print_warning "Mismatch between WAV and LAB files (WAV: ${WAV_COUNT}, LAB: ${LAB_COUNT})"
fi

print_status "Found ${WAV_COUNT} WAV files and ${LAB_COUNT} LAB files"

# Create necessary directories
mkdir -p "${TEMP_CORPUS}"
mkdir -p "${MFA_DIR}"
mkdir -p "${OOVS_DIR}"
mkdir -p "${TEMP_OUT_ALIGN}"

# Create temporary corpus structure for MFA (speaker-organized)
print_status "Creating temporary corpus structure for MFA..."

# Extract speaker IDs from filenames and organize by speaker
cd "${PREPARED_DIR}"
for wav_file in *.wav; do
    if [ -f "$wav_file" ]; then
        base_name="${wav_file%.wav}"
        lab_file="${base_name}.lab"
        
        if [ -f "$lab_file" ]; then
            # Extract speaker ID from filename (first part before first hyphen)
            speaker_id="${base_name%%-*}"
            
            # Create speaker directory
            speaker_dir="${TEMP_CORPUS}/${DATASET_NAME}_${speaker_id}"
            mkdir -p "${speaker_dir}"
            
            # Copy files to speaker directory
            cp "$wav_file" "${speaker_dir}/"
            cp "$lab_file" "${speaker_dir}/"
        else
            print_warning "No corresponding LAB file for ${wav_file}"
        fi
    fi
done

# Count organized files
ORGANIZED_COUNT=$(find "${TEMP_CORPUS}" -name "*.wav" | wc -l)
print_status "Organized ${ORGANIZED_COUNT} files into speaker directories"

if [ "${ORGANIZED_COUNT}" -eq 0 ]; then
    print_error "No files were organized for MFA"
    exit 1
fi

# Activate MFA environment and run the pipeline
print_status "Activating MFA environment..."

# Use eval with micromamba to properly activate the environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate mfa

# Check if MFA is available in the activated environment
if ! command_exists mfa; then
    print_error "MFA not found in the activated environment"
    print_error "Please ensure montreal-forced-aligner is installed in the mfa environment"
    exit 1
fi

print_status "MFA environment activated successfully"

# Step 1: Find OOVs from the corpus
print_status "Step 1: Finding OOVs from corpus..."
mfa find_oovs "${TEMP_CORPUS}" \
              english_us_arpa \
              "${OOVS_DIR}" || {
    print_warning "OOV finding failed, but continuing with alignment"
}

# Step 2: Generate pronunciations for OOVs using G2P
OOVS_FILE="${OOVS_DIR}/oovs_found_english_us_arpa.txt"
OOVS_DICT="${OOVS_DIR}/oovs.dict"

if [ -f "${OOVS_FILE}" ] && [ -s "${OOVS_FILE}" ]; then
    print_status "Step 2: Generating pronunciations for OOVs..."
    mfa g2p "${OOVS_FILE}" \
            english_us_arpa \
            "${OOVS_DICT}" || {
        print_warning "G2P failed, but continuing with alignment"
    }
    
    # Step 3: Add pronunciations to the ARPA dictionary model
    if [ -f "${OOVS_DICT}" ] && [ -s "${OOVS_DICT}" ]; then
        print_status "Step 3: Adding pronunciations to ARPA dictionary..."
        mfa model add_words english_us_arpa "${OOVS_DICT}" || {
            print_warning "Adding words to dictionary failed, but continuing with alignment"
        }
    fi
else
    print_status "No OOVs found, proceeding with existing dictionary"
fi

# Step 4: Perform forced alignment
print_status "Step 4: Performing forced alignment..."
print_status "This may take a while depending on your corpus size..."

mfa align "${TEMP_CORPUS}" \
          english_us_arpa \
          english_us_arpa \
          "${TEMP_OUT_ALIGN}" \
          --clean \
          --output_format json

if [ $? -eq 0 ]; then
    print_success "Forced alignment completed successfully!"
    
    # Step 5: Copy alignment results back to prepared dataset
    print_status "Step 5: Copying alignment results to prepared dataset..."
    
    ALIGNED_COUNT=0
    # Find all JSON files in speaker subdirectories (skip alignment_analysis.csv)
    for json_file in $(find "${TEMP_OUT_ALIGN}" -name "*.json" -not -name "alignment_analysis*"); do
        base_name=$(basename "$json_file" .json)
        dest_file="${PREPARED_DIR}/${base_name}.json"
        
        if [ -f "${PREPARED_DIR}/${base_name}.wav" ]; then
            cp "$json_file" "$dest_file"
            ((ALIGNED_COUNT++))
        else
            print_warning "No corresponding WAV file for alignment: ${base_name}"
        fi
    done
    
    print_success "Copied ${ALIGNED_COUNT} alignment files to prepared dataset"
    
    # Display some statistics
    print_status "Alignment statistics:"
    print_status "  - Original files: ${WAV_COUNT}"
    print_status "  - Aligned files: ${ALIGNED_COUNT}"
    if [ "${ALIGNED_COUNT}" -lt "${WAV_COUNT}" ]; then
        print_warning "  - Missing alignments: $((WAV_COUNT - ALIGNED_COUNT))"
    fi
    
else
    print_error "Forced alignment failed"
    exit 1
fi

# Cleanup temporary files
print_status "Cleaning up temporary files..."
rm -rf "${TEMP_CORPUS}"
rm -rf "${TEMP_OUT_ALIGN}"
rm -rf "${MFA_DIR}"

print_success "MFA alignment pipeline completed successfully!"
print_success "Dataset ${DATASET_NAME} now has alignment data"
print_status "Prepared dataset location: ${PREPARED_DIR}"
