# OpenLipSync

An open-source, cross-platform project that converts audio input into realistic facial expressions in real-time following the [MPEG-4 (FBA)](https://visagetechnologies.com/uploads/2012/08/MPEG-4FBAOverview.pdf) standard.

## Setup for model training

Core (uv)

```bash
uv init
uv sync                  
```

MFA (micromamba)

```bash
micromamba create -n mfa -c conda-forge python=3.12 montreal-forced-aligner
micromamba activate mfa

mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa
mfa model download g2p english_us_arpa
```

Run `python training/scripts/downloadTrainingData.py` to download training samples.

Run `python training/scripts/createDataCorpus.py` to convert LibriSpeech data to training format.

Run `./run_mfa_alignment.sh` to perform forced alignment.
