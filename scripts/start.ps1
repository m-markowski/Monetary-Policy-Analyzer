$ErrorActionPreference = 'Stop'
$ProgressPreference = 'Continue'
$launcherLock = $null
$exitCode = 1

try {
    $root = Split-Path -Parent $PSScriptRoot
    Set-Location -LiteralPath $root
    Write-Host "`nMonetary Policy Analyzer"
    Write-Host 'Setup is local to this repository. No administrator access is requested.'
    Write-Host 'On first use, you will be asked to paste your FRED API key.'

    $architecture = $env:PROCESSOR_ARCHITECTURE
    if ($env:PROCESSOR_ARCHITEW6432) { $architecture = $env:PROCESSOR_ARCHITEW6432 }
    if ($architecture -ne 'AMD64' -or [Environment]::OSVersion.Version.Major -lt 10) {
        throw 'This launcher requires Windows 10/11, 64-bit, on an Intel/AMD processor. Windows ARM is not supported.'
    }
    if ($PSVersionTable.PSVersion -lt [version]'5.1') {
        throw 'Windows PowerShell 5.1 or newer is required.'
    }
    foreach ($name in @('pyproject.toml', 'scripts\bootstrap.py', 'app\Main_Page.py', 'config\config.yaml')) {
        if (-not (Test-Path -LiteralPath (Join-Path $root $name) -PathType Leaf)) {
            throw "Missing $name. Copy the launcher into the root of the complete repository."
        }
    }
    foreach ($name in @('.runtime', '.venv')) {
        $path = Join-Path $root $name
        if (Test-Path -LiteralPath $path) {
            $item = Get-Item -LiteralPath $path -Force
            if (-not $item.PSIsContainer -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
                throw "$name must be a real directory inside this repository, not a file or directory link."
            }
        }
    }

    $runtime = Join-Path $root '.runtime'
    New-Item -ItemType Directory -Force -Path $runtime | Out-Null
    try {
        $launcherLock = [IO.File]::Open(
            (Join-Path $runtime 'launcher.lock'), [IO.FileMode]::OpenOrCreate,
            [IO.FileAccess]::ReadWrite, [IO.FileShare]::None
        )
    } catch [IO.IOException] {
        throw 'Another launcher may already be running for this repository. Use its window or close it first.'
    }

    # Keep settings from other Python environments out of this process.
    Get-ChildItem Env: | Where-Object { $_.Name -match '^(UV_|PIP_|PYTHON)' } | ForEach-Object {
        Remove-Item -LiteralPath "Env:$($_.Name)"
    }
    foreach ($name in @('VIRTUAL_ENV', 'CONDA_PREFIX', 'CONDA_DEFAULT_ENV', 'TF_USE_LEGACY_KERAS')) {
        Remove-Item -LiteralPath "Env:$name" -ErrorAction SilentlyContinue
    }
    $uvVersion = '0.12.10'
    $env:UV_UNMANAGED_INSTALL = Join-Path $runtime 'uv'
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $runtime 'python'
    $env:UV_PYTHON_BIN_DIR = Join-Path $runtime 'bin'
    $env:UV_CACHE_DIR = Join-Path $runtime 'uv-cache'
    $env:UV_PYTHON_INSTALL_REGISTRY = 'false'
    $env:UV_PYTHON_INSTALL_BIN = 'false'
    $env:UV_PYTHON_NO_REGISTRY = 'true'
    $env:UV_MANAGED_PYTHON = 'true'
    $env:UV_NO_SYSTEM_CONFIG = 'true'
    $env:UV_HTTP_TIMEOUT = '120'
    $env:UV_HTTP_RETRIES = '3'
    $env:TEMP = Join-Path $runtime 'tmp'
    $env:TMP = $env:TEMP
    New-Item -ItemType Directory -Force -Path $env:UV_UNMANAGED_INSTALL, $env:TEMP | Out-Null
    $OutputEncoding = [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
    $uv = Join-Path $env:UV_UNMANAGED_INSTALL 'uv.exe'

    $installedVersion = ''
    if (Test-Path -LiteralPath $uv -PathType Leaf) {
        try {
            $installedVersion = & $uv --version
            if ($LASTEXITCODE -ne 0) { $installedVersion = '' }
        } catch { $installedVersion = '' }
    }
    if ($installedVersion -notmatch "^uv $([regex]::Escape($uvVersion))( |$)") {
        Write-Host "`n[1/5] Downloading uv $uvVersion into .runtime\uv..."
        [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
        $installer = Join-Path $env:TEMP 'install-uv.ps1'
        for ($attempt = 1; $attempt -le 3; $attempt++) {
            try {
                Invoke-WebRequest -UseBasicParsing -Uri "https://astral.sh/uv/$uvVersion/install.ps1" -OutFile $installer -TimeoutSec 120
                break
            } catch {
                if ($attempt -eq 3) { throw 'Could not download uv. Check access to astral.sh and run start.bat again.' }
                Write-Host "Download failed. Retrying ($attempt/3)..."
                Start-Sleep -Seconds 2
            }
        }
        & "$PSHOME\powershell.exe" -NoLogo -NoProfile -ExecutionPolicy Bypass -File $installer
        if ($LASTEXITCODE -ne 0) { throw 'The uv installer failed. Review its output and check your network connection.' }
        Remove-Item -LiteralPath $installer -Force
        $installedVersion = & $uv --version
        if ($LASTEXITCODE -ne 0 -or $installedVersion -notmatch "^uv $([regex]::Escape($uvVersion))( |$)") {
            throw 'The downloaded uv executable could not be verified by its version.'
        }
    } else {
        Write-Host "`n[1/5] Local uv is ready."
    }
    Write-Host $installedVersion

    Write-Host "`n[2/5] Checking the repository-local Python 3.12..."
    $savedPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $python = & $uv python find --no-config --system --managed-python --no-python-downloads 3.12 2>$null
    $found = $LASTEXITCODE -eq 0
    $ErrorActionPreference = $savedPreference
    if (-not $found) {
        Write-Host 'Downloading Python. Existing global Python installations will not be changed.'
        & $uv python install --no-config --no-bin --no-registry 3.12
        if ($LASTEXITCODE -ne 0) { throw 'Python could not be downloaded. Check the output above and run start.bat again.' }
        $python = & $uv python find --no-config --system --managed-python --no-python-downloads 3.12
        if ($LASTEXITCODE -ne 0) { throw 'The repository-local Python could not be found.' }
    }
    $python = "$python".Trim()
    $localPythonDirectory = [IO.Path]::GetFullPath($env:UV_PYTHON_INSTALL_DIR).TrimEnd('\') + '\'
    if (-not [IO.Path]::GetFullPath($python).StartsWith($localPythonDirectory, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Refusing to use Python from outside this repository.'
    }
    & $python -I -X utf8 --version
    if ($LASTEXITCODE -ne 0) { throw 'The local Python executable could not be started.' }
    & $python -I -X utf8 (Join-Path $PSScriptRoot 'bootstrap.py') @args
    $exitCode = $LASTEXITCODE
} catch {
    Write-Host "`nERROR: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host 'No global Python installation is needed. Fix the reported problem and run start.bat again.'
    $exitCode = 1
} finally {
    if ($null -ne $launcherLock) { $launcherLock.Dispose() }
}
exit $exitCode
