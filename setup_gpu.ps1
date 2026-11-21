# Install CUDA-enabled PyTorch into .venv (Windows, NVIDIA GPU).
#   powershell -ExecutionPolicy Bypass -File setup_gpu.ps1
# 1. reads the driver's max CUDA version from nvidia-smi
# 2. picks the newest PyTorch CUDA build that driver supports (cu130 / cu128; RTX 50-series needs >= cu128)
# 3. installs torch from that index into .venv, plus requirements.txt
# 4. checks torch.cuda.is_available() and prints the GPU name
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { python -m venv .venv }

# nvidia-smi ships with the driver in System32 (or NVSMI on old drivers); PATH may miss both
$smi = (Get-Command nvidia-smi -ErrorAction SilentlyContinue).Source
if (-not $smi) {
    $smi = @("$env:SystemRoot\System32\nvidia-smi.exe",
             "$env:ProgramFiles\NVIDIA Corporation\NVSMI\nvidia-smi.exe") |
           Where-Object { Test-Path $_ } | Select-Object -First 1
}
if (-not $smi) {
    Write-Host "nvidia-smi not found: the NVIDIA driver is not installed. Install the GeForce driver, then rerun." -ForegroundColor Yellow
    Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion | Format-Table
    exit 1
}
& $smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
# Pick the CUDA build from the driver version (the nvidia-smi header format changes between
# drivers). Windows minimums from NVIDIA's CUDA release notes: 13.0 -> 580.88, 12.8 -> 570.65.
$drv = [version]((& $smi --query-gpu=driver_version --format=csv,noheader | Select-Object -First 1).Trim())
Write-Host "driver $drv"
if     ($drv -ge [version]"580.88") { $tag = "cu130" }
elseif ($drv -ge [version]"570.65") { $tag = "cu128" }
else { Write-Host "Driver too old: RTX 50-series GPUs need a CUDA >= 12.8 build of PyTorch (driver >= 570.65). Update the NVIDIA driver, then rerun." -ForegroundColor Yellow; exit 1 }
Write-Host "installing torch ($tag) into .venv"

& $py -m pip install --upgrade pip
& $py -m pip install torch --index-url "https://download.pytorch.org/whl/$tag"
& $py -m pip install -r requirements.txt
& $py -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"
