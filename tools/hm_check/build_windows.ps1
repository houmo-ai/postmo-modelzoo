<#
build_windows.ps1
PowerShell 脚本，用于在 Windows (MSVC) 下以 CMake + Visual Studio 构建 hm-check。
用法示例：
  .\build_windows.ps1 -BuildDir "build" -Configuration Release -Generator "Visual Studio 17 2022" -TCIM "C:\tcim" -HOUMO "C:\houmo_sdk" -Install
#>
param(
    [string]$SourceDir = "${PSScriptRoot}",
    [string]$BuildDir = "${PSScriptRoot}\build",
    [ValidateSet("Release","Debug")][string]$Configuration = "Release",
    [string]$Generator = "Visual Studio 17 2022",
    [string]$Platform = "x64",
    [string]$TCIM = "",
    [string]$HOUMO = "",
    [switch]$NoInstall  # Changed from Install to NoInstall
)

function Check-Command {
    param([string]$cmd)
    $null = Get-Command $cmd -ErrorAction SilentlyContinue
    return $?
}

if (-not (Check-Command cmake)) {
    Write-Error "CMake is not found in PATH. Please install CMake and add it to PATH."
    exit 1
}

if ($TCIM -ne "") { $env:TCIM_RUNTIME_PATH = $TCIM }
if ($HOUMO -ne "") { $env:HOUMO_SDK_PATH = $HOUMO }

if (-not $env:TCIM_RUNTIME_PATH) {
    Write-Error "Environment variable TCIM_RUNTIME_PATH is not set. Provide via -TCIM or set it in system environment."
    exit 1
}
if (-not $env:HOUMO_SDK_PATH) {
    Write-Error "Environment variable HOUMO_SDK_PATH is not set. Provide via -HOUMO or set it in system environment."
    exit 1
}

Write-Host "Source: $SourceDir"
Write-Host "Build dir: $BuildDir"
Write-Host "Generator: $Generator"
Write-Host "Platform: $Platform"
Write-Host "Configuration: $Configuration"
Write-Host "TCIM_RUNTIME_PATH: $env:TCIM_RUNTIME_PATH"
Write-Host "HOUMO_SDK_PATH: $env:HOUMO_SDK_PATH"

if (-not (Test-Path $BuildDir)) { New-Item -ItemType Directory -Path $BuildDir | Out-Null }

# Configure
$cmakeArgs = @(
    '-S', $SourceDir,
    '-B', $BuildDir,
    '-G', $Generator,
    '-A', $Platform,
    "-DCMAKE_BUILD_TYPE=$Configuration"
)

# Add install prefix for default installation
$installDir = Join-Path -Path $SourceDir -ChildPath "..\bin"
$cmakeArgs += "-DCMAKE_INSTALL_PREFIX=$installDir"

$configureCmd = "cmake " + ($cmakeArgs -join ' ')
Write-Host "Running: $configureCmd"
$rc = & cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) { Write-Error "cmake configuration failed"; exit $LASTEXITCODE }

# Build
$buildArgs = @('--build', $BuildDir, '--config', $Configuration, '--', '/m')
$buildCmd = "cmake " + ($buildArgs -join ' ')
Write-Host "Running: $buildCmd"
& cmake @buildArgs
if ($LASTEXITCODE -ne 0) { Write-Error "cmake build failed"; exit $LASTEXITCODE }

# Optional install step - now enabled by default
if (-not $NoInstall) {
    $installDir = Join-Path -Path $SourceDir -ChildPath "..\bin"
    if (-not (Test-Path $installDir)) { New-Item -ItemType Directory -Path $installDir | Out-Null }
    $installArgs = @('--install', $BuildDir, '--config', $Configuration, '--prefix', $installDir)
    Write-Host "Installing to: $installDir"
    & cmake @installArgs
    if ($LASTEXITCODE -ne 0) { Write-Error "cmake install failed"; exit $LASTEXITCODE }
}

Write-Host "Build complete. Executable located in: $BuildDir\$Configuration\hm-check.exe (or $BuildDir\hm-check.exe for single-config generators)"
Write-Host "Installation (if enabled) available in: $(Join-Path -Path $SourceDir -ChildPath "..\bin")"