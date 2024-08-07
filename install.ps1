<#
.SYNOPSIS
    A script to run GPUStack server or worker.

.DESCRIPTION
    Run GPUStack server or worker with the specified settings.

.EXAMPLE
    .\install.ps1

    You can start the GPUStack server by running the command on a server node.

.EXAMPLE
    .\install.ps1 --server-url http://myserver --token mytoken

    You can add additional workers to form a GPUStack cluster by running the command on worker nodes.
#>

$ErrorActionPreference = "Stop"

$INSTALL_PACKAGE_SPEC = if ($env:INSTALL_PACKAGE_SPEC) { $env:INSTALL_PACKAGE_SPEC } else { "gpustack" }
$INSTALL_PRE_RELEASE = if ($env:INSTALL_PRE_RELEASE) { $env:INSTALL_PRE_RELEASE } else { 0 }
$INSTALL_INDEX_URL = if ($env:INSTALL_INDEX_URL) { $env:INSTALL_INDEX_URL } else { "" }

function Log-Info {
    param (
        [Parameter(Position = 0)]
        [string]$message
    )
    Write-Host "[INFO] $message"
}

function Log-Warn {
    param (
        [Parameter(Position = 0)]
        [string]$message
    )
    Write-Host "[WARN] $message" -ForegroundColor Yellow
}

function Log-Fatal {
    param (
        [Parameter(Position = 0)]
        [string]$message
    )
    Write-Host "[ERROR] $message" -ForegroundColor Red
}

# Function to check if the script is run as administrator
function Check-AdminPrivilege {
    if (-NOT ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
        throw "This script must be run as Administrator. Please restart the script with Administrator privileges."
    }
}

# Function to detect the OS
function Check-OS {
    $OS = (Get-CimInstance -Class Win32_OperatingSystem).Caption
    if ($OS -notmatch "Windows") {
        throw "Unsupported OS. Only Windows is supported."
    }
}

function Check-CUDA {
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        if (-not (Get-Command nvcc -ErrorAction SilentlyContinue)) {
            throw "NVIDIA GPU detected but CUDA is not installed. Please install CUDA."
        }
    }
}

function Get-Arg {
    param (
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$RemainingArgs
    )

    Log-Info "Getting arguments from flags..."

    $envList = @()

    for ($i = 0; $i -lt $RemainingArgs.Count; $i++) {
        $value = $RemainingArgs[$i + 1]
        switch ($RemainingArgs[$i]) {
            "--debug" {
                if ($value -eq "False" -or $value -eq "false") {
                    $envList += "GPUSTACK_DEBUG=False"
                }
                else {
                    $envList += "GPUSTACK_DEBUG=True"
                }
            }
            "--config-file" {
                $envList += "GPUSTACK_CONFIG_File=$value"
                $i++
            }
            "--data-dir" {
                $envList += "GPUSTACK_DATA_DIR=$value"
                $i++
            }
            "--token" {
                $envList += "GPUSTACK_TOKEN=$value"
                $i++
            }
            "-t" {
                $envList += "GPUSTACK_TOKEN=$value"
                $i++
            }
            "--host" {
                $envList += "GPUSTACK_HOST=$value"
                $i++
            }
            "--port" {
                $envList += "GPUSTACK_PORT=$value"
                $i++
            }
            "--database-url" {
                $envList += "GPUSTACK_DATABASE_URL=$value"
                $i++
            }
            "--bootstrap-password" {
                $envList += "GPUSTACK_BOOTSTRAP_PASSWORD=$value"
                $i++
            }
            "--disable-worker" {
                if ($value -eq "False" -or $value -eq "false") {
                    $envList += "GPUSTACK_DISABLE_WORKER=False"
                }
                else {
                    $envList += "GPUSTACK_DISABLE_WORKER=True"
                }
            }
            "--system-reserved" {
                $escapedJsonString = $value -replace '"', '\`"'
                $envList += "GPUSTACK_SYSTEM_RESERVED=`"$escapedJsonString`""
                $i++
            }
            "--ssl-keyfile" {
                $envList += "GPUSTACK_SSL_KEY_FILE=$value"
                $i++
            }
            "--ssl-certfile" {
                $envList += "GPUSTACK_SSL_CERT_FILE=$value"
                $i++
            }
            "--force-auth-localhost" {
                if ($value -eq "False" -or $value -eq "false") {
                    $envList += "GPUSTACK_FORCE_AUTH_LOCALHOST=False"
                }
                else {
                    $envList += "GPUSTACK_FORCE_AUTH_LOCALHOST=True"
                }
                $i++
            }
            "--server-url" {
                $envList += "GPUSTACK_SERVER_URL=$value"
                $i++
            }
            "-s" {
                $envList += "GPUSTACK_SERVER_URL=$value"
                $i++
            }
            "--worker-ip" {
                $envList += "GPUSTACK_WORKER_IP=$value"
                $i++
            }
            "--worker-port" {
                $envList += "GPUSTACK_WORKER_PORT=$value"
                $i++
            }
            "--enable-metrics" {
                if ($value -eq "False" -or $value -eq "false") {
                    $envList += "GPUSTACK_ENABLE_METRICS=False"
                }
                else {
                    $envList += "GPUSTACK_ENABLE_METRICS=True"
                }
            }
            "--metrics-port" {
                $envList += "GPUSTACK_METRICS_PORT=$value"
                $i++
            }
            "--log-dir" {
                $envList += "GPUSTACK_LOG_DIR=$value"
                $i++
            }
        }
    }


    $envList += "APPDATA=$env:APPDATA"

    $envListString = $envList -join " "

    return $envListString
}

function Refresh-ChocolateyProfile {
    $chocoInstallPath = [System.Environment]::GetEnvironmentVariable("ChocolateyInstall", "Machine")
    if (-not $chocoInstallPath) {
        throw "Chocolatey installation path not found. Ensure Chocolatey is installed correctly."
    }

    $chocoHelpersPath = Join-Path -Path $chocoInstallPath -ChildPath "helpers\chocolateyProfile.psm1"
    try {
        Import-Module $chocoHelpersPath -ErrorAction Stop
        if (-not (Get-Command refreshenv -ErrorAction SilentlyContinue)) {
            throw "Could not find 'refreshenv'. Something is wrong with Chocolatey installation."
        }

        refreshenv
    }
    catch {
        throw "Failed to import Chocolatey profile. Ensure Chocolatey is installed correctly."
    }
}

function Install-Chocolatey {
    if (Get-Command choco -ErrorAction SilentlyContinue) {
        Log-Info "Chocolatey already installed."
        return
    }

    try {
        Log-Info "Installing Chocolatey..."
        Set-ExecutionPolicy Bypass -Scope Process -Force
        Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))
        Refresh-ChocolateyProfile
        Log-Info "Chocolatey installed successfully."
    }
    catch {
        throw "Failed to install Chocolatey: `"$($_.Exception.Message)`""
    }
}

function Install-Python {

    $needInstallPython = $true
    $PYTHON_VERSION = $null
    if (Get-Command python -ErrorAction SilentlyContinue) {
        $PYTHON_VERSION = python -c 'import sys; print(sys.version_info.major * 10 + sys.version_info.minor)'
        $pythonSource = $(Get-Command python).Source
        $isDirty = (($null -eq $PYTHON_VERSION) -or ($PYTHON_VERSION -eq "")) -and ($pythonSource -match "WindowsApps")

        if ($isDirty) {
            Log-Info "Python command is just alias for open Windows Store, clean it up..."
            Remove-Item "$env:LOCALAPPDATA\Microsoft\WindowsApps\python.exe" -ErrorAction SilentlyContinue
            Remove-Item "$env:LOCALAPPDATA\Microsoft\WindowsApps\python3.exe" -ErrorAction SilentlyContinue
        }
        elseif ($PYTHON_VERSION -lt 40) {
            throw "Python version is $PYTHON_VERSION, which is less than 3.10. Please upgrade Python to at least version 3.10."
        }
        else {
            $needInstallPython = $false
            Log-Info "Python already installed."
        }
    }

    if ($needInstallPython) {
        try {
            Log-Info "Installing Python..."
            $null = choco install python --version=3.10.11 -y
            Refresh-ChocolateyProfile
            Log-Info "Python installed successfully."
        }
        catch {
            throw "Failed to install Python: `"$($_.Exception.Message)`""
        }
    }

    if (-not (Get-Command pipx -ErrorAction SilentlyContinue)) {
        try {
            Log-Info "Pipx could not be found. Attempting to install..."

            python -m pip install pipx
            if ($LASTEXITCODE -ne 0) {
                throw "failed to install pipx."
            }

            pipx ensurepath
            if ($LASTEXITCODE -ne 0) {
                throw "failed to run pipx ensurepath."
            }

            Log-Info "Pipx installed successfully."
        }
        catch {
            throw "Failed to install Pipx: `"$($_.Exception.Message)`""
        }
    }
    else {
        Log-Info "Pipx already installed."
    }
}

function Install-NSSM {
    if (Get-Command nssm -ErrorAction SilentlyContinue) {
        Log-Info "NSSM already installed."
        return
    }

    try {
        Log-Info "Installing NSSM..."
        choco install nssm -y
        if ($LASTEXITCODE -ne 0) {
            throw "failed to install nssm."
        }

        Refresh-ChocolateyProfile
        if ($LASTEXITCODE -ne 0) {
            throw "failed to refresh chocolatey profile."
        }

        Log-Info "NSSM installed successfully."
    }
    catch {
        throw "Failed to install NSSM: `"$($_.Exception.Message)`""
    }
}

function Install-GPUStack {
    $action = "Install"
    if (Get-Command gpustack -ErrorAction SilentlyContinue) {
        $action = "Upgrade"
        Log-Info "GPUStack already installed, Upgrading..."
    }

    try {
        Log-Info "$action GPUStack..."
        $installArgs = @()
        if ($INSTALL_PRE_RELEASE -eq 1) {
            $installArgs += "--pip-args='--pre'"
        }

        if ($INSTALL_INDEX_URL) {
            $installArgs += "--index-url=$INSTALL_INDEX_URL"
        }

        Log-Info "$action GPUStack with $($installArgs -join ' ') $INSTALL_PACKAGE_SPEC"

        $pythonPath = Get-Command python | Select-Object -ExpandProperty Source
        $env:PIPX_DEFAULT_PYTHON = $pythonPath

        Log-Info "Check pipx environment..."
        $pipxSharedEnv = (pipx environment --value PIPX_SHARED_LIBS)
        if ($LASTEXITCODE -ne 0) {
            throw "failed to run pipx environment --value PIPX_SHARED_LIBS."
        }

        $pipxSharedConfigPath = (Join-Path -Path $pipxSharedEnv -ChildPath "pyvenv.cfg")
        if (Test-Path $pipxSharedConfigPath) {
            $configContent = Get-Content -Path (Join-Path -Path $pipxSharedEnv -ChildPath "pyvenv.cfg")
            $homeValue = ""
            foreach ($line in $configContent) {
                if ($line.StartsWith("home =")) {
                    $homeValue = $line.Split("=")[1].Trim()
                    break
                }
            }
            if (-not (Test-Path -Path $homeValue)) {
                Log-Warn "Current pipx config is invalid with isn't exist python path $homeValue, try to refresh shared environment..."
                python -m venv --clear $pipxSharedEnv
                if ($LASTEXITCODE -ne 0) {
                    throw "failed to refresh virtual environment."
                }
            }
        }

        Log-Info "$action GPUStack with pipx and pythin $pythonPath..."
        if ($action -ieq "Upgrade") {
            Log-Info "Uninstall existing gpustack..."

            Stop-GPUStackService

            pipx uninstall gpustack
            if ($LASTEXITCODE -ne 0) {
                throw "failed to uninstall existing gpustack."
            }
        }

        pipx install --force --verbose @installArgs $INSTALL_PACKAGE_SPEC
        if ($LASTEXITCODE -ne 0) {
            throw "failed to install $INSTALL_PACKAGE_SPEC."
        }

        pipx ensurepath
        if ($LASTEXITCODE -ne 0) {
            throw "failed to run pipx ensurepath."
        }

        Log-Info "Updating PATH environment variable..."

        $pipEnv = (pipx environment --value PIPX_BIN_DIR)
        if ($LASTEXITCODE -ne 0) {
            throw "failed to run pipx environment."
        }

        $env:Path = "$pipEnv;$env:Path"

        $currentPath = [Environment]::GetEnvironmentVariable("PATH", "Machine")
        if (!$currentPath.Contains($pipEnv)) {
            [Environment]::SetEnvironmentVariable("Path", $env:Path, [EnvironmentVariableTarget]::Machine)
        }
        else {
            Log-Info "Path already contains $pipEnv"
        }

        Log-Info "$action GPUStack success."
    }
    catch {
        throw "Failed to $action GPUStack: `"$($_.Exception.Message)`""
    }
}

function Stop-GPUStackService {
    # Check if the service already exists.
    $serviceName = "GPUStack"
    $gpustack = Get-Service -Name $serviceName -ErrorAction SilentlyContinue
    if ($null -ne $gpustack) {
        try {
            Log-Info "Stopping existing ${serviceName} service..."
            $result = nssm stop $serviceName confirm
            if ($LASTEXITCODE -eq 0) {
                Log-Info "Stopped existing ${serviceName} success"
            }
            else {
                Log-Warn "Failed to stop existing ${serviceName} service: `"$($result)`""
            }

            $result = nssm remove $serviceName confirm
            if ($LASTEXITCODE -eq 0) {
                Log-Info "Removed existing ${serviceName} success"
            }
            else {
                Log-Warn "Failed to remove existing ${serviceName} service: `"$($result)`""
            }
        }
        catch {
            throw "Failed to stop and remove existing ${serviceName} service: `"$($_.Exception.Message)`""
        }
    }
}

function Setup-GPUStackService {
    param (
        [string]$envListString
    )

    $serviceName = "GPUStack"
    $serviceDisplayName = "GPUStack"
    $exePath = $(Get-Command gpustack).Source
    $exeFile = Get-Item -Path $exePath

    if ($exeFile.Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
        $exePath = (Get-Item -Path $exeFile).Target
    }

    try {
        Stop-GPUStackService

        Log-Info "Creating ${serviceName} service..."

        $appDataPath = $env:APPDATA
        $gpustackDirectoryName = "gpustack"
        $gpustackDirectoryPath = Join-Path -Path $appDataPath -ChildPath $gpustackDirectoryName

        $gpustackLogDirectoryPath = Join-Path -Path $gpustackDirectoryPath -ChildPath "log"
        $gpustackLogPath = Join-Path -Path $gpustackLogDirectoryPath -ChildPath "gpustack.log"

        $null = New-Item -Path $gpustackDirectoryPath -ItemType "Directory" -ErrorAction SilentlyContinue -Force
        $null = New-Item -Path $gpustackLogDirectoryPath -ItemType "Directory" -ErrorAction SilentlyContinue -Force

        $null = nssm install $serviceName $exePath
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to install service $serviceName"
        }

        $commands = @(
            "nssm set $serviceName AppDirectory $gpustackDirectoryPath",
            "nssm set $serviceName AppParameters 'start'",
            "nssm set $serviceName DisplayName $serviceDisplayName",
            "nssm set $serviceName Description 'GPUStack aims to get you started with managing GPU devices, running LLMs and performing inference in a simple yet scalable manner.'",
            "nssm set $serviceName Start SERVICE_AUTO_START",
            "nssm set $serviceName ObjectName LocalSystem",
            "nssm set $serviceName AppExit Default Restart",
            "nssm set $serviceName AppStdout $gpustackLogPath",
            "nssm set $serviceName AppStderr $gpustackLogPath",
            "nssm set $serviceName AppEnvironmentExtra $envListString"
        )

        foreach ($cmd in $commands) {
            $null = Invoke-Expression $cmd
            if ($LASTEXITCODE -ne 0) { throw "Failed to run nssm set environment: $cmd" }
        }

        Log-Info "Starting ${serviceName} service..."
        $null = nssm start $serviceName -y

        # Wait for the service to start for 120 seconds.
        $startTime = Get-Date
        while ((nssm status $serviceName) -ne 'SERVICE_RUNNING' -and ((Get-Date) - $startTime).TotalSeconds -lt 120) {
            Log-Info "Waiting for $serviceName service to start."
            Start-Sleep -s 5
        }
        if ((nssm status $serviceName) -eq 'SERVICE_RUNNING') {
            Log-Info "${serviceName} service created and started successfully."
        }
        else {
            Log-Info "$serviceName log:"
            Get-Content -Path $gpustackLogPath -Tail 300

            Log-Info "$serviceName service dump:"
            nssm dump GPUStack
            if ($LASTEXITCODE -ne 0) {
                throw "Failed to dump service $serviceName"
            }
        }
    }
    catch {
        throw "Failed to setup ${serviceName}: `"$($_.Exception.Message)`""
    }
}


function Create-UninstallScript {
    $gpustackDirectoryName = "gpustack"

    $gpustacUninstallScriptContent = @'
<#
.SYNOPSIS
    A script to uninstall GPUStack server or worker.

.DESCRIPTION
    Uninstall GPUStack server or worker with the specified settings.

.EXAMPLE
    .\uninstall.ps1

    You can uninstall the runing GPUStack server/worker by running the command.
#>

[CmdletBinding()]

param (
    [switch]$Help
)

$ErrorActionPreference = "Stop"

if ($Help) {
    Get-Help -Full $MyInvocation.MyCommand.Definition
    exit
}

function Log-Info {
    param (
        [Parameter(Position = 0)]
        [string]$message
    )
    Write-Host "[INFO] $message"
}

function Log-Warn {
    param (
        [Parameter(Position = 0)]
        [string]$message
    )
    Write-Host "[WARN] $message" -ForegroundColor Yellow
}

function Log-Fatal {
    param (
        [Parameter(Position = 0)]
        [string]$message
    )
    Write-Host "[ERROR] $message" -ForegroundColor Red
}

# Function to check if the script is run as administrator
function Check-AdminPrivilege {
    if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
        throw "This script must be run as Administrator. Please restart the script with Administrator privileges."
    }
}

# Function to detect the OS
function Check-OS {
    $OS = (Get-CimInstance -Class Win32_OperatingSystem).Caption
    if ($OS -notmatch "Windows") {
        throw "Unsupported OS. Only Windows is supported."
    }
}

function Uninstall-GPUStack {
    $serviceName = "GPUStack"
    $packageName = "gpustack"
    try {
        Log-Info "Stopping GPUStack..."
        $gpustack = Get-Service -Name $serviceName -ErrorAction SilentlyContinue
        if ($null -ne $gpustack) {
            try {
                Log-Info "Stopping existing ${serviceName} service..."
                $result = nssm stop $serviceName confirm
                if ($LASTEXITCODE -eq 0) {
                    Log-Info "Stopped ${serviceName} success"
                }
                else {
                    Log-Warn "Failed to stop existing ${serviceName} service: `"$($result)`""
                }

                $result = nssm remove $serviceName confirm
                if ($LASTEXITCODE -eq 0) {
                    Log-Info "Removed ${serviceName} success"
                }
                else {
                    Log-Warn "Failed to remove existing ${serviceName} service: `"$($result)`""
                }

            }
            catch {
                throw "Failed to stop and remove existing ${serviceName} service: `"$($_.Exception.Message)`""
            }
        }else{
            Log-Info "No existing ${serviceName} service found."
        }


        if (-not(Get-Command pipx -ErrorAction SilentlyContinue)) {
            throw "Pipx not found."
        }

        Log-Info "Uninstalling package ${packageName}..."
        $pipxPackages = pipx list
        if ($pipxPackages -like '*gpustack*') {
            pipx uninstall gpustack
            Log-Info "Uninstalled package ${packageName} success."
        }
        else {
            Log-Info "Package ${packageName} is not installed."
        }

        Log-Info "Cleaning up..."
        $appDataPath = $env:APPDATA
        $gpustackDirectoryPath = Join-Path -Path $appDataPath -ChildPath $packageName
        if (Test-Path -Path $gpustackDirectoryPath) {
            Get-ChildItem -Path $appDataPath -Filter $packageName | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
    catch {
        throw "Failed to uninstall GPUStack: `"$($_.Exception.Message)`""
    }
}

try {
    Check-AdminPrivilege
    Check-OS
    Uninstall-GPUStack
}
catch {
    Log-Fatal "Failed to uninstall GPUStack: `"$($_.Exception.Message)`""
}
'@
    try {
        Log-Info "Creating uninstall script..."

        $appDataPath = $env:APPDATA
        $gpustackDirectoryPath = Join-Path -Path $appDataPath -ChildPath $gpustackDirectoryName
        $null = New-Item -ItemType Directory -Path $gpustackDirectoryPath -ErrorAction SilentlyContinue -Force

        $gpustacUninstallScriptPath = Join-Path -Path $gpustackDirectoryPath -ChildPath "uninstall.ps1"
        if (Test-Path -Path $gpustacUninstallScriptPath) {
            Log-Info "Removing existing uninstall script $gpustackUninstallScriptPath ..."
            Remove-Item -Path $gpustacUninstallScriptPath -Force -ErrorAction SilentlyContinue
        }

        Set-Content -Path $gpustacUninstallScriptPath -Value $gpustacUninstallScriptContent
        Log-Info "Uninstall script created successfully at $gpustacUninstallScriptPath"
    }
    catch {
        Log-Warn "Failed to create uninstall script: `"$($_.Exception.Message)`""
    }
}


try {
    Check-AdminPrivilege
    Check-OS
    Check-CUDA
    Install-Chocolatey
    Install-Python
    Install-NSSM
    Install-GPUStack
    Create-UninstallScript
    $envListString = Get-Arg @args
    Setup-GPUStackService -envListString $envListString
}
catch {
    Log-Fatal "Failed to install GPUStack: `"$($_.Exception.Message)`""
}
