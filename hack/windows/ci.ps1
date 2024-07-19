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
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    & make lint @ciArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    & make test @ciArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    & make validate @ciArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    & make build @ciArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

}

#
# main
#

GPUStack.Log.Info "+++ CI +++"
try {
    Invoke-CI $args
} catch {
    GPUStack.Log.Fatal "failed run ci: $($_.Exception.Message)"
}
GPUStack.Log.Info "--- CI ---"
