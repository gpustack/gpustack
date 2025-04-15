# Set error handling
$ErrorActionPreference = "Stop"

# Get the root directory and third_party directory
$ROOT_DIR = Split-Path -Path $MyInvocation.MyCommand.Definition -Parent | Split-Path -Parent | Split-Path -Parent -Resolve

# Include the common functions
. "$ROOT_DIR/hack/lib/windows/init.ps1"

function Install-Dependency {
    pip install poetry==1.8.3
    if ($LASTEXITCODE -ne 0) {
        GPUStack.Log.Fatal "failed to install poetry."
    }

    poetry install
    if ($LASTEXITCODE -ne 0) {
        GPUStack.Log.Fatal "failed run poetry install."
    }

    poetry run pre-commit install
    if ($LASTEXITCODE -ne 0) {
        GPUStack.Log.Fatal "failed run pre-commint install."
    }
}

function Get-UI {
    $defaultTag = "latest"
    $uiPath = Join-Path -Path $ROOT_DIR -ChildPath "gpustack/ui"
    $tmpPath = Join-Path -Path $uiPath -ChildPath "tmp"
    $tmpUIPath = Join-Path -Path $tmpPath -ChildPath "ui"
    $tag = "latest"

    $null = Remove-Item -Recurse -Force $uiPath -ErrorAction Ignore
    $null = New-Item -ItemType Directory -Path $tmpUIPath

    GPUStack.Log.Info "downloading UI assets"

    try {
        $tmpFile = "$tmpPath/ui.tar.gz"
        $url = "https://gpustack-ui-1303613262.cos.accelerate.myqcloud.com/releases/$tag.tar.gz"
        DownloadWithRetries -url $url -outFile $tmpFile -maxRetries 3

        # For git action's bug, can't use tar directly.
        # https://github.com/julia-actions/setup-julia/issues/205
        & "$env:WINDIR/System32/tar" -xzf "$tmpPath/ui.tar.gz" -C "$tmpUIPath"
    }
    catch {
        GPUStack.Log.Fatal "failed to download '$tag' UI archive: $($_.Exception.Message)"

        if (-eq $tag $defaultTag) {
            return
        }

        GPUStack.Log.Warn "failed to download '$tag' UI archive, fallback to '$defaultTag' UI archive"

        try {
            $tmpFile = "$tmpPath/ui.tar.gz"
            $url = "https://gpustack-ui-1303613262.cos.accelerate.myqcloud.com/releases/$defaultTag.tar.gz"
            DownloadWithRetries -url $url -outFile $tmpFile -maxRetries 3
            tar -xzf $tmpFile -C "$tmpUIPath"
        }
        catch {
            GPUStack.Log.Fatal "failed to download '$defaultTag' UI archive: : $($_.Exception.Message)"
        }
    }

    Copy-Item -Path "$tmpUIPath/dist/*" -Destination $uiPath -Recurse
    Remove-Item -Recurse -Force $tmpUIPath -ErrorAction Ignore
}

function DownloadWithRetries {
    param (
        [string]$url,
        [string]$outFile,
        [int]$maxRetries = 3
    )

    for ($i = 1; $i -le $maxRetries; $i++) {
        try {
            GPUStack.Log.Info "Attempting to download from $url (Attempt $i of $maxRetries)"
            Invoke-WebRequest -Uri $url -OutFile $outFile -ErrorAction Stop
            return
        }
        catch {
            GPUStack.Log.Warn "Download attempt $i failed: $($_.Exception.Message)"
            if ($i -eq $maxRetries) {
                throw $_
            }
        }
    }
}

#
# main
#

GPUStack.Log.Info "+++ DEPENDENCIES +++"
try {
    Install-Dependency
    Get-UI
}
catch {
    GPUStack.Log.Fatal "failed to download dependencies: $($_.Exception.Message)"
}
GPUStack.Log.Info "-- DEPENDENCIES ---"
