$ErrorActionPreference = "Stop"

# Get the root directory and third_party directory
$ROOT_DIR = Split-Path -Path $MyInvocation.MyCommand.Definition -Parent | Split-Path -Parent | Split-Path -Parent -Resolve

# Include the common functions
. "$ROOT_DIR/hack/lib/windows/init.ps1"

function Validate {
    param (
        [string]$file
    )

    # Generate the current checksum
    $currentChecksum = Get-FileHash -Path $file -Algorithm SHA256
    $generatedChecksumFile = "${file}.generated.sha256sum"
    $currentChecksum.Hash | Out-File -FilePath $generatedChecksumFile

    # Compare with the expected checksum
    $expectedChecksumFile = "${file}.sha256sum"
    $expectedChecksum = Get-Content -Path $expectedChecksumFile

    if ($currentChecksum.Hash -eq $expectedChecksum.Trim()) {
        GPUStack.Log.Info "Checksums match."
        Remove-Item -Path $generatedChecksumFile
    }
    else {
        GPUStack.Log.Fatal "Checksums do not match!`nPlease run 'Get-FileHash -Path ${file} -Algorithm SHA256 | Select -ExpandProperty Hash > ${file}.sha256sum' to update the checksum."
    }
}

#
# main
#

GPUStack.Log.Info "+++ VALIDATE +++"
try {
    Validate -file "install.ps1"
}
catch {
    GPUStack.Log.Fatal "failed to test: $($_.Exception.Message)"
}
GPUStack.Log.Info "--- VALIDATE ---"
