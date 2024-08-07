$ErrorActionPreference = "Stop"

# Get the root directory and third_party directory
$ROOT_DIR = Split-Path -Path $MyInvocation.MyCommand.Definition -Parent | Split-Path -Parent | Split-Path -Parent -Resolve

# Include the common functions
. "$ROOT_DIR/hack/lib/windows/init.ps1"

function Get-PSScriptAnalyzer {
    $module = Get-Module -ListAvailable -Name PSScriptAnalyzer
    if (-not $module) {
        Install-Module -Name PSScriptAnalyzer -Scope CurrentUser -Force -SkipPublisherCheck -AllowClobber
    }
}

function Lint {
    param (
        [string]$path
    )

    GPUStack.Log.Info "linting $path"

    $result = Invoke-ScriptAnalyzer -Path $ROOT_DIR -Recurse -EnableExit -ExcludeRule PSUseBOMForUnicodeEncodedFile,PSAvoidUsingPlainTextForPassword,PSAvoidUsingInvokeExpression, PSReviewUnusedParameter, PSUseApprovedVerbs, PSAvoidGlobalVars, PSUseShouldProcessForStateChangingFunctions, PSAvoidUsingWriteHost
    $result | Format-Table -AutoSize
    if ($result.Length -ne 0) {
        GPUStack.Log.Fatal "failed with Invoke-ScriptAnalyzer lint."
    }

    poetry run pre-commit run flake8 --all-files
    if ($LASTEXITCODE -ne 0) {
        GPUStack.Log.Fatal "failed with flake8 lint."
    }
    poetry run pre-commit run black --all-files
    if ($LASTEXITCODE -ne 0) {
        GPUStack.Log.Fatal "failed with black lint."
    }
    poetry run pre-commit run check-yaml --all-files
    if ($LASTEXITCODE -ne 0) {
        GPUStack.Log.Fatal "failed with check-yaml lint."
    }
    poetry run pre-commit run debug-statements --all-files
    if ($LASTEXITCODE -ne 0) {
        GPUStack.Log.Fatal "failed with debug-statements lint."
    }
    poetry run pre-commit run end-of-file-fixer --all-files
    if ($LASTEXITCODE -ne 0) {
        GPUStack.Log.Fatal "failed with end-of-file-fixer lint."
    }
}

function Remove-BOM {
    $filePath = Join-Path $ROOT_DIR -ChildPath "install.ps1"

    $bytes = [System.IO.File]::ReadAllBytes($filePath)
    if ($bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF) {
        [System.IO.File]::WriteAllBytes($filePath, $bytes[3..($bytes.Length - 1)])
    }
    Write-Host "BOM removed from $filePath"
}

#
# main
#

GPUStack.Log.Info "+++ LINT +++"
try {
    Get-PSScriptAnalyzer
    Lint "gpustack"
    Remove-BOM
}
catch {
    GPUStack.Log.Fatal "failed to lint: $($_.Exception.Message)"
}
GPUStack.Log.Info "--- LINT ---"
