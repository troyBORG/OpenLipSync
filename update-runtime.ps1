Set-Location G:\git\OpenLipSync
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m pip install uv
uv lock --upgrade-package torchcodec
uv export --locked --no-hashes --output-file recommended.txt
python -m pip install -r .\recommended.txt --extra-index-url https://download.pytorch.org/whl/cu128 -U