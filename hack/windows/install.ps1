# Set error handling
$ErrorActionPreference = "Stop"

# Get the root directory and third_party directory
$ROOT_DIR = Split-Path -Path $MyInvocation.MyCommand.Definition -Parent | Split-Path -Parent | Split-Path -Parent -Resolve

# Include the common functions
. "$ROOT_DIR/hack/lib/windows/init.ps1"

function Install-Dependency {
    pip install uv
    if ($LASTEXITCODE -ne 0) {
        GPUStack.Log.Fatal "failed to install uv."
    }

    uv sync
    if ($LASTEXITCODE -ne 0) {
        GPUStack.Log.Fatal "failed run uv sync."
    }

    uv run pre-commit install
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

    if ($GIT_VERSION -ne "v0.0.0") {
        $tag = $GIT_VERSION
    }

    $null = Remove-Item -Recurse -Force $uiPath -ErrorAction Ignore
    $null = New-Item -ItemType Directory -Path $tmpUIPath

    GPUStack.Log.Info "downloading '$tag' UI assets"

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

function Package-Chart {
    # The chart is served to a registered cluster from the UI's static tree, so
    # it has to be packaged into the wheel the same way the Unix build does it.
    # Without this the wheel builds and installs cleanly and then refuses every
    # manifest request, since `chart_available()` finds nothing to serve.
    #
    # After Get-UI, never before: that step removes the whole ui directory.
    if ($env:CHART_PACKAGE -eq "false") {
        GPUStack.Log.Info "skipping Helm chart packaging"
        return
    }

    $chartPath = Join-Path -Path $ROOT_DIR -ChildPath "charts/gpustack-chart"
    $targetDir = Join-Path -Path $ROOT_DIR -ChildPath "gpustack/ui/static/charts"

    if (-not (Get-Command "helm" -ErrorAction Ignore)) {
        GPUStack.Log.Fatal "helm is required to package the chart; install it or set CHART_PACKAGE=false"
    }

    GPUStack.Log.Info "packaging Helm chart"

    # Vendor only what the pins are not already satisfied by, so a build box
    # that has them stays offline and a moved pin is not packaged from a stale
    # archive still sitting there.
    #
    # A failed listing is not an empty one: without the exit-code check it reads
    # as "all satisfied" and the build packages whichever archive happens to be
    # under charts/ — the case this check exists to catch.
    $listed = @(helm dependency list $chartPath 2>$null)
    if ($LASTEXITCODE -ne 0) {
        GPUStack.Log.Info "could not read the chart's dependencies, vendoring them"
        $unsatisfied = @("(unknown)")
    }
    else {
        $unsatisfied = @($listed |
            Select-Object -Skip 1 |
            Where-Object { $_.Trim() -ne "" -and ($_ -split '\s+')[-1] -ne "ok" })
    }
    if ($unsatisfied.Count -gt 0) {
        GPUStack.Log.Info "vendoring the chart's dependencies"
        helm dependency update $chartPath | Out-Null
        if ($LASTEXITCODE -ne 0) {
            GPUStack.Log.Fatal "failed to vendor the chart's dependencies"
        }
    }

    $null = Remove-Item -Recurse -Force $targetDir -ErrorAction Ignore
    $null = New-Item -ItemType Directory -Path $targetDir -Force

    helm package $chartPath --destination $targetDir | Out-Null
    if ($LASTEXITCODE -ne 0) {
        GPUStack.Log.Fatal "failed to package the chart"
    }

    # A fixed file name, not the chart's version: the manifest that references it
    # is regenerated by the same server that serves it.
    $packaged = Get-ChildItem -Path $targetDir -Filter "gpustack-chart-*.tgz" | Select-Object -First 1
    if ($null -eq $packaged) {
        GPUStack.Log.Fatal "failed to package the chart"
    }
    Move-Item -Path $packaged.FullName -Destination (Join-Path -Path $targetDir -ChildPath "gpustack-chart.tgz") -Force

    GPUStack.Log.Info "packaged Helm chart"
}

#
# main
#

GPUStack.Log.Info "+++ DEPENDENCIES +++"
try {
    Install-Dependency
    Get-UI
    Package-Chart
}
catch {
    GPUStack.Log.Fatal "failed to download dependencies: $($_.Exception.Message)"
}
GPUStack.Log.Info "-- DEPENDENCIES ---"
