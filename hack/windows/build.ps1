# Set error handling
$ErrorActionPreference = "Stop"

$ROOT_DIR = Split-Path -Path $MyInvocation.MyCommand.Definition -Parent | Split-Path -Parent | Split-Path -Parent -Resolve

# Include the common functions
. "$ROOT_DIR/hack/lib/windows/init.ps1"

function Build {
    $distDir = Join-Path -Path $ROOT_DIR -ChildPath "dist"
    Remove-Item -Path $distDir -Recurse -Force -ErrorAction SilentlyContinue

    poetry build
    if ($LASTEXITCODE -ne 0) {
        GPUStack.Log.Fatal "failed to run poetry build."
    }

    $whlFiles = Get-ChildItem -Path $distDir -Filter "*.whl" -File
    if ($whlFiles.Count -eq 0) {
        GPUStack.Log.Fatal "no wheel files found in $distDir"
    }

    foreach ($whlFile in $whlFiles) {
        $orginalName = $whlFile.Name
        $newName = $orginalName -replace "any", "win_amd64"

        $newFilePath = Join-Path -Path $distDir -ChildPath $newName
        Remove-Item -Path $newFilePath -Force -ErrorAction SilentlyContinue
        Rename-Item -Path $whlFile.FullName -NewName $newFilePath -Force
        GPUStack.Log.Info "renamed $orginalName to $newName"
    }
}

function Install-Dependency {
    & "$ROOT_DIR\hack\windows\install.ps1"
}

function Set-Version {
    $versionFile = Join-Path -Path $ROOT_DIR -ChildPath "gpustack\__init__.py"
    $version = if ($null -ne $global:GIT_VERSION) { $global:GIT_VERSION } else { "v0.0.0" }
    $gitCommit = if ($null -ne $global:GIT_COMMIT) { $global:GIT_COMMIT } else { "HEAD" }
    $gitCommitShort = $gitCommit.Substring(0, [Math]::Min(7, $gitCommit.Length))

    GPUStack.Log.Info "setting version to $version"
    GPUStack.Log.Info "setting git commit to $gitCommitShort"

    # Replace the __version__ variable in the __init__.py file
    $fileContent = Get-Content -Path $versionFile
    $fileContent = $fileContent -replace "__version__ = .*", "__version__ = '$version'"
    $fileContent = $fileContent -replace "__git_commit__ = .*", "__git_commit__ = '$gitCommitShort'"
    Set-Content -Path $versionFile -Value $fileContent

    # Update the poetry version
    poetry version "$version"
}

function Restore-Version-File {
    $versionFile = Join-Path -Path $ROOT_DIR -ChildPath "gpustack\__init__.py"

    git checkout -- $versionFile
    if ($LASTEXITCODE -ne 0) {
        GPUStack.Log.Fatal "failed restore version file."
    }
}

#
# main
#

GPUStack.Log.Info "+++ BUILD +++"
try {
    Install-Dependency
    Set-Version
    Build
    Restore-Version-File
}
catch {
    GPUStack.Log.Fatal "failed to build: $($_.Exception.Message)"
}
GPUStack.Log.Info "--- BUILD ---"
