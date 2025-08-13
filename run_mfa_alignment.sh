#!/bin/bash

# OpenLipSync MFA Alignment Script
# This script runs the complete MFA alignment pipeline:
# 1. Find OOVs (Out-of-Vocabulary words)
# 2. Generate pronunciations for OOVs using G2P
# 3. Add new words to the dictionary
# 4. Perform forced alignment

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
    print_error "micromamba create -n mfa -c conda-forge python=3.12 montreal-forced-alignment"
    exit 1
fi

# Define paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_CORPUS="${SCRIPT_DIR}/training/data/data_corpus"
MFA_DIR="${SCRIPT_DIR}/training/data/_mfa"
OOVS_DIR="${MFA_DIR}/oovs"
OUT_ALIGN="${SCRIPT_DIR}/training/data/out_align"

# Create necessary directories
mkdir -p "${MFA_DIR}"
mkdir -p "${OOVS_DIR}"
mkdir -p "${OUT_ALIGN}"

print_status "Starting MFA alignment pipeline..."
print_status "Data corpus: ${DATA_CORPUS}"
print_status "MFA directory: ${MFA_DIR}"
print_status "Output directory: ${OUT_ALIGN}"

# Check if data corpus exists
if [ ! -d "${DATA_CORPUS}" ]; then
    print_error "Data corpus directory not found: ${DATA_CORPUS}"
    print_error "Please ensure you have run the data preparation scripts first."
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
    print_error "Please ensure montreal-forced-alignment is installed in the mfa environment"
    exit 1
fi

print_status "MFA environment activated successfully"

# Step 1: Find OOVs from the corpus
print_status "Step 1: Finding OOVs from corpus..."
mfa find_oovs "${DATA_CORPUS}" \
              english_us_arpa \
              "${OOVS_DIR}"

if [ $? -eq 0 ]; then
    print_success "OOVs found successfully"
else
    print_error "Failed to find OOVs"
    exit 1
fi

# Step 2: Generate pronunciations for OOVs using G2P
OOVS_FILE="${OOVS_DIR}/oovs_found_english_us_arpa.txt"
OOVS_DICT="${OOVS_DIR}/oovs.dict"

if [ -f "${OOVS_FILE}" ]; then
    print_status "Step 2: Generating pronunciations for OOVs..."
    mfa g2p "${OOVS_FILE}" \
            english_us_arpa \
            "${OOVS_DICT}"
    
    if [ $? -eq 0 ]; then
        print_success "G2P pronunciations generated successfully"
    else
        print_error "Failed to generate G2P pronunciations"
        exit 1
    fi
else
    print_warning "No OOVs file found at ${OOVS_FILE}"
    print_warning "This might mean there are no out-of-vocabulary words in your corpus"
fi

# Step 3: Add pronunciations to the ARPA dictionary model
if [ -f "${OOVS_DICT}" ]; then
    print_status "Step 3: Adding pronunciations to ARPA dictionary..."
    mfa model add_words english_us_arpa "${OOVS_DICT}"
    
    if [ $? -eq 0 ]; then
        print_success "Words added to dictionary successfully"
    else
        print_error "Failed to add words to dictionary"
        exit 1
    fi
else
    print_warning "No OOVs dictionary found at ${OOVS_DICT}"
    print_warning "Proceeding with alignment using existing dictionary"
fi

# Step 4: Perform forced alignment
print_status "Step 4: Performing forced alignment..."
print_status "This may take a while depending on your corpus size..."

mfa align "${DATA_CORPUS}" \
          english_us_arpa \
          english_us_arpa \
          "${OUT_ALIGN}" \
          --clean \
          --output_format json

if [ $? -eq 0 ]; then
    print_success "Forced alignment completed successfully!"
    print_success "Results saved to: ${OUT_ALIGN}"
    
    # Display some statistics
    if [ -d "${OUT_ALIGN}" ]; then
        NUM_FILES=$(find "${OUT_ALIGN}" -name "*.json" | wc -l)
        print_status "Generated ${NUM_FILES} alignment files"
    fi
else
    print_error "Forced alignment failed"
    exit 1
fi

print_success "MFA alignment pipeline completed successfully!"
print_status "You can find the alignment results in: ${OUT_ALIGN}"
