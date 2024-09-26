# Set error handling
$ErrorActionPreference = "Stop"

# Get the root directory and third_party directory
$ROOT_DIR = Split-Path -Path $MyInvocation.MyCommand.Definition -Parent | Split-Path -Parent | Split-Path -Parent -Resolve
$THIRD_PARTY_DIR = Join-Path -Path $ROOT_DIR -ChildPath "gpustack/third_party/bin"

# Include the common functions
. "$ROOT_DIR/hack/lib/windows/init.ps1"

function Install-Dependency {
    pip install poetry==1.7.1
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

function Get-FastFetch {
    $version = "2.25.0.1"
    $fastfetchDir = Join-Path -Path $THIRD_PARTY_DIR -ChildPath "fastfetch"
    $fastfetchTmpDir = Join-Path -Path $fastfetchDir -ChildPath "tmp"

    # Include more platforms if needed
    $platforms = @("windows-amd64")

    foreach ($platform in $platforms) {
        $targetFile = Join-Path -Path $fastfetchDir -ChildPath "fastfetch-$platform.exe"

        if (Test-Path -Path $targetFile) {
            GPUStack.Log.Info "fastfetch-$platform already exists, skipping download"
            continue
        }

        GPUStack.Log.Info "downloading fastfetch-$platform '$version' archive"

        try {
            $tmpFile = Join-Path -Path $fastfetchTmpDir -ChildPath "fastfetch-$platform.zip"
            Remove-Item -Recurse -Force $fastfetchTmpDir -ErrorAction Ignore
            New-Item -ItemType Directory -Path $fastfetchTmpDir | Out-Null

            $url = "https://github.com/gpustack/fastfetch/releases/download/$version/fastfetch-$platform.zip"
            DownloadWithRetries -url $url -outFile $tmpFile -maxRetries 3
            Expand-Archive -Path $tmpFile -DestinationPath $fastfetchTmpDir

            $tmpBinFile = Join-Path -Path $fastfetchTmpDir -ChildPath "fastfetch.exe"

            Copy-Item -Path $tmpBinFile -Destination $targetFile
        }
        catch {
            GPUStack.Log.Fatal "failed to download fastfetch-$platform '$version' archive: : $($_.Exception.Message)"
        }

    }

    Remove-Item -Recurse -Force $fastfetchTmpDir -ErrorAction Ignore
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

function Get-GGUFParser {
    $version = "v0.11.0"
    $ggufParserDir = Join-Path -Path $THIRD_PARTY_DIR -ChildPath "gguf-parser"
    New-Item -ItemType Directory -Path $ggufParserDir -ErrorAction Ignore | Out-Null

    # Include more platforms if needed
    $platforms = @("windows-amd64","windows-arm64")

    foreach ($platform in $platforms) {
        $targetBinFile = "gguf-parser-$platform.exe"

        $targetFile = Join-Path -Path $ggufParserDir -ChildPath $targetBinFile
        if (Test-Path -Path $targetFile) {
            GPUStack.Log.Info "gguf-parser-$platform already exists, skipping download"
            continue
        }

        GPUStack.Log.Info "downloading gguf-parser-$platform '$version' archive"
        try {
            $url = "https://github.com/gpustack/gguf-parser-go/releases/download/$version/$targetBinFile"
            DownloadWithRetries -url $url -outFile $targetFile -maxRetries 3
        }
        catch {
            GPUStack.Log.Fatal "failed to download gguf-parser-$platform '$version' archive: $($_.Exception.Message)"
        }
    }
}

function Get-LlamaBox {
    $version = "v0.0.50"
    $llamaBoxDir = Join-Path -Path $THIRD_PARTY_DIR -ChildPath "llama-box"
    $llamaBoxTmpDir = Join-Path -Path $llamaBoxDir -ChildPath "tmp"

    # Include more platforms if needed
    $platforms = @("windows-amd64-cuda-12.6", "windows-amd64-avx2", "windows-arm64-neon")

    foreach ($platform in $platforms) {
        $binFile = "llama-box.exe"

        $targetFile = Join-Path -Path $llamaBoxDir -ChildPath "llama-box-$platform.exe"

        if (Test-Path -Path $targetFile) {
            GPUStack.Log.Info "llama-box-$platform already exists, skipping download"
            continue
        }

        GPUStack.Log.Info "downloading llama-box-$platform '$version' archive"
        try {
            $llamaBoxPlatformTmpDir = Join-Path -Path $llamaBoxTmpDir -ChildPath $platform
            Remove-Item -Recurse -Force $llamaBoxPlatformTmpDir -ErrorAction Ignore
            New-Item -ItemType Directory -Path $llamaBoxPlatformTmpDir | Out-Null

            $tmpFile = Join-Path -Path $llamaBoxTmpDir -ChildPath "llama-box-$version-$platform.zip"
            $url = "https://github.com/gpustack/llama-box/releases/download/$version/llama-box-$platform.zip"
            DownloadWithRetries -url $url -outFile $tmpFile -maxRetries 4

            Expand-Archive -Path $tmpFile -DestinationPath $llamaBoxPlatformTmpDir
            Copy-Item -Path "$llamaBoxPlatformTmpDir/$binFile" -Destination $targetFile
        }
        catch {
            GPUStack.Log.Fatal "failed to download llama-box-$platform '$version' archive: $($_.Exception.Message)"
        }
    }

    Remove-Item -Recurse -Force $llamaBoxTmpDir -ErrorAction Ignore
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
    Get-FastFetch
    Get-GGUFParser
    Get-LlamaBox
    Get-UI
}
catch {
    GPUStack.Log.Fatal "failed to download dependencies: $($_.Exception.Message)"
}
GPUStack.Log.Info "-- DEPENDENCIES ---"
