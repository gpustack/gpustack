$ErrorActionPreference = "Stop"

# Get the root directory and third_party directory
$ROOT_DIR = Split-Path -Path $MyInvocation.MyCommand.Definition -Parent | Split-Path -Parent | Split-Path -Parent -Resolve

# Include the common functions
. "$ROOT_DIR/hack/lib/windows/init.ps1"

function Generate {
    poetry run gen
}

#
# main
#

GPUStack.Log.Info "+++ GENERATE +++"
try {
    Generate
} catch {
    GPUStack.Log.Fatal "failed to generate: $($_.Exception.Message)"
}
GPUStack.Log.Info "--- GENERATE ---"
