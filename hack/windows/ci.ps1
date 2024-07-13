# Set error handling
$ErrorActionPreference = "Stop"

# Get the root directory and third_party directory
$ROOT_DIR = Split-Path -Path $MyInvocation.MyCommand.Definition -Parent | Split-Path -Parent | Split-Path -Parent -Resolve

# Include the common functions
. "$ROOT_DIR/hack/lib/windows/init.ps1"

function Invoke-CI {
    param(
        [string[]]$ciArgs
    )

    & make install @ciArgs
    & make lint @ciArgs
    & make test @ciArgs
    & make validate @ciArgs
    & make build @ciArgs
}

#
# main
#

GPUStack.Log.Info "+++ CI +++"
Invoke-CI $args
GPUStack.Log.Info "--- CI ---"
