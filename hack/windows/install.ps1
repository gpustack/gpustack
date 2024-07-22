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
    $version = "2.18.1.1"
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

            Invoke-WebRequest -Uri "https://github.com/aiwantaozi/fastfetch/releases/download/$version/fastfetch-$platform.zip" -OutFile $tmpFile -UseBasicParsing
            Expand-Archive -Path $tmpFile -DestinationPath $fastfetchTmpDir

            $tmpBinFile = Join-Path -Path $fastfetchTmpDir -ChildPath "fastfetch.exe"

            Copy-Item -Path $tmpBinFile -Destination $targetFile
        } catch {
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
        Invoke-WebRequest -Uri "https://gpustack-ui-1303613262.cos.accelerate.myqcloud.com/releases/$tag.tar.gz" -OutFile "$tmpPath/ui.tar.gz" -UseBasicParsing

        # For git action's bug, can't use tar directly.
        # https://github.com/julia-actions/setup-julia/issues/205
        & "$env:WINDIR/System32/tar" -xzf "$tmpPath/ui.tar.gz" -C "$tmpUIPath"
    } catch {
        GPUStack.Log.Fatal "failed to download '$tag' UI archive: $($_.Exception.Message)"

        if (-eq $tag $defaultTag) {
            return
        }

        GPUStack.Log.Warn "failed to download '$tag' UI archive, fallback to '$defaultTag' UI archive"

        try {
            Invoke-WebRequest -Uri "https://gpustack-ui-1303613262.cos.accelerate.myqcloud.com/releases/$defaultTag.tar.gz" -OutFile "$tmpPath/ui.tar.gz" -UseBasicParsing
            tar -xzf "$tmpPath/ui.tar.gz" -C "$tmpUIPath"
        } catch {
            GPUStack.Log.Fatal "failed to download '$defaultTag' UI archive: : $($_.Exception.Message)"
        }
    }

    Copy-Item -Path "$tmpUIPath/dist/*" -Destination $uiPath -Recurse
    Remove-Item -Recurse -Force $tmpUIPath -ErrorAction Ignore
}

function Get-GGUFParser {
    $version = "v0.3.2"
    $ggufParserDir = Join-Path -Path $THIRD_PARTY_DIR -ChildPath "gguf-parser"
    New-Item -ItemType Directory -Path $ggufParserDir -ErrorAction Ignore | Out-Null

    # Include more platforms if needed
    $platforms = @("windows-amd64")

    foreach ($platform in $platforms) {
        $targetBinFile = "gguf-parser-$platform.exe"

        $targetFile = Join-Path -Path $ggufParserDir -ChildPath $targetBinFile
        if (Test-Path -Path $targetFile) {
            GPUStack.Log.Info "gguf-parser-$platform already exists, skipping download"
            continue
        }

        GPUStack.Log.Info "downloading gguf-parser-$platform '$version' archive"
        try {
            Invoke-WebRequest -Uri "https://github.com/thxCode/gguf-parser-go/releases/download/$version/$targetBinFile" -OutFile $targetFile -UseBasicParsing
            # chmod +x $targetFile
        } catch {
            GPUStack.Log.Fatal "failed to download gguf-parser-$platform '$version' archive: $($_.Exception.Message)"
        }
    }
}

function Get-LlamaBox {
    $version = "v0.0.13"
    $llamaBoxDir = Join-Path -Path $THIRD_PARTY_DIR -ChildPath "llama-box"
    $llamaBoxTmpDir = Join-Path -Path $llamaBoxDir -ChildPath "tmp"

    # Include more platforms if needed
    $platforms = @("windows-amd64-cuda-12.5-s")

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
            Invoke-WebRequest -Uri "https://github.com/thxCode/llama-box/releases/download/$version/llama-box-$platform.zip" -OutFile $tmpFile -UseBasicParsing

            Expand-Archive -Path $tmpFile -DestinationPath $llamaBoxPlatformTmpDir
            Copy-Item -Path "$llamaBoxPlatformTmpDir/$binFile" -Destination $targetFile
        }
        catch {
            GPUStack.Log.Fatal "failed to download llama-box-$platform '$version' archive: $($_.Exception.Message)"
        }
    }

    Remove-Item -Recurse -Force $llamaBoxTmpDir -ErrorAction Ignore
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
} catch {
    GPUStack.Log.Fatal "failed to download dependencies: $($_.Exception.Message)"
}
GPUStack.Log.Info "-- DEPENDENCIES ---"
