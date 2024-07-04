$ErrorActionPreference = "Stop"

# Get the root directory and third_party directory
$ROOT_DIR = Split-Path -Path $MyInvocation.MyCommand.Definition -Parent | Split-Path -Parent | Split-Path -Parent -Resolve

# Include the common functions
. "$ROOT_DIR/hack/lib/windows/init.ps1"

function Test {
    poetry run pytest
}

#
# main
#

GPUStack.Log.Info "+++ TEST +++"
try {
    Test
} catch {
    GPUStack.Log.Fatal "failed to test: $($_.Exception.Message)"
}
GPUStack.Log.Info "--- TEST ---"
