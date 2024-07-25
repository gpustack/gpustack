# Set error handling
$ErrorActionPreference = "Stop"

# Get the root directory and third_party directory
$ROOT_DIR = Split-Path -Path $MyInvocation.MyCommand.Definition -Parent | Split-Path -Parent | Split-Path -Parent -Resolve

# Include the common functions
. "$ROOT_DIR/hack/lib/windows/init.ps1"

function Publish-Pypi {
    poetry run twine check dist/*.whl
    poetry run twine upload dist/*.whl
    if ($LASTEXITCODE -ne 0) {
        GPUStack.Log.Fatal "twine upload failed."
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
