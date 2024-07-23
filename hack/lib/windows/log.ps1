# Logger variables helpers. These functions need the following variables:
# LOG_LEVEL  -  The level of logger, default is "debug".
$log_level = $env:LOG_LEVEL -or "debug"
$log_colorful = $env:LOG_COLORFUL -or $true

function GPUStack.Log.Errexit {
    trap {
        $ErrorMessage = $_.Exception.Message
        $ErrorCode = $_.Exception.HResult

        GPUStack.Log.Panic -message $ErrorMessage -code $ErrorCode
        continue
    }
}

# Debug level logging.
function GPUStack.Log.Debug {
    param (
        [Parameter(Position = 0)]
        [string]$message
    )
    if ($log_level -ne "debug") {
        return
    }
    $timestamp = Get-Timestamp
    Write-Output "[$timestamp] [DEBG] $message"
}

# Info level logging.
function GPUStack.Log.Info {
    param (
        [Parameter(Position = 0)]
        [string]$message
    )
    if ($log_level -ne "debug" -and $log_level -ne "info") {
        return
    }
    $timestamp = Get-Timestamp
    if ($log_colorful -eq $true) {
        Write-Host "[$timestamp] " -NoNewline
        Write-Host "[INFO] " -NoNewline -ForegroundColor Blue
        Write-Host $message
    }
    else {
        Write-Output "[$timestamp] [INFO] $message"
    }
}

# Warn level logging.
function GPUStack.Log.Warn {
    param (
        [Parameter(Position = 0)]
        [string]$message
    )
    $timestamp = Get-Timestamp
    if ($log_colorful -eq $true) {
        Write-Host "[$timestamp] " -NoNewline
        Write-Host "[WARN] " -NoNewline -ForegroundColor Yellow
        Write-Host $message
    }
    else {
        Write-Output "[$timestamp] [WARN] $message"
    }
}

# Error level logging, log an error but keep going, don't dump the stack or exit.
function GPUStack.Log.Error {
    param (
        [Parameter(Position = 0)]
        [string]$message
    )
    $timestamp = Get-Timestamp
    if ($log_colorful -eq $true) {
        Write-Host "[$timestamp] " -NoNewline
        Write-Host "[ERRO] " -NoNewline -ForegroundColor Red
        Write-Host $message
    }
    else {
        Write-Output "[$timestamp] [ERRO] $message"
    }
}

# Fatal level logging, log an error but exit with 1, don't dump the stack or exit.
function GPUStack.Log.Fatal {
    param (
        [Parameter(Position = 0)]
        [string]$message
    )
    $timestamp = Get-Timestamp
    if ($log_colorful -eq $true) {
        Write-Host "[$timestamp] " -NoNewline
        Write-Host "[FATA] " -NoNewline -ForegroundColor Red
        Write-Host $message
    }
    else {
        Write-Output "[$timestamp] [FATA] $message"
    }
    exit 1
}

# Panic level logging, dump the error stack and exit.
function GPUStack.Log.Panic {
    param (
        [string]$message,
        [int]$code = 1
    )
    $timestamp = Get-Timestamp
    $formattedMessage = "[$timestamp] [FATA] $message"
    if ($log_colorful -eq $true) {
        Write-Error $formattedMessage
    } else {
        Write-Output $formattedMessage
    }
    $customStackTrace = [System.Diagnostics.StackTrace]::new()
    $customStackTrace.GetFrames() | ForEach-Object {
        Write-Error ("       {0}: {1}" -f $_.GetFileName(), $_.GetFileLineNumber())
    }
    exit $code
}

function Get-Timestamp {
    return $(Get-Date -Format 'MMdd HH:mm:ss')
}
