# Set error handling
$ErrorActionPreference = "Stop"

# Get the root directory and third_party directory
$ROOT_DIR = Split-Path -Path $MyInvocation.MyCommand.Definition -Parent | Split-Path -Parent | Split-Path -Parent -Resolve

# Include the common functions
. "$ROOT_DIR/hack/lib/windows/init.ps1"

function Publish-Pypi {
    if (-not $env:PYPI_API_TOKEN) {
        GPUStack.Log.Fatal "PYPI_API_TOKEN is not set"
    }

    poetry publish --username __token__ --password $env:PYPI_API_TOKEN
    if ($LASTEXITCODE -ne 0) {
        GPUStack.Log.Fatal "failed to run poetry publish."
    }
}

#
# main
#

GPUStack.Log.Info "+++ Publish Pypi +++"
try {
    Publish-Pypi
}
catch {
    GPUStack.Log.Fatal "failed to publish Pypi: $($_.Exception.Message)"
}
GPUStack.Log.Info "--- Publish Pypi ---"
