$ErrorActionPreference = "Stop"
$null = $PSModuleAutoLoadingPreference

# Set no_proxy for localhost if behind a proxy, otherwise,
# the connections to localhost in scripts will time out.
$env:no_proxy = "127.0.0.1,localhost"

$ROOT_DIR = Split-Path -Path $MyInvocation.MyCommand.Definition -Parent | Split-Path -Parent | Split-Path -Parent | Split-Path -Parent -Resolve

Get-ChildItem -Path "$ROOT_DIR/hack/lib/windows" -File | ForEach-Object {
    if ($_.Name -ne "init.ps1") {
        . $_.FullName
    }
}

GPUStack.Log.Errexit
Get-GPUStackVersionVar
