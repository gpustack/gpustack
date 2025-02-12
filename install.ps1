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

# Script updated at: 2025-02-19T08:16:49Z

$ErrorActionPreference = "Stop"

$INSTALL_PACKAGE_SPEC = if ($env:INSTALL_PACKAGE_SPEC) { $env:INSTALL_PACKAGE_SPEC } else { "gpustack[audio]" }
$INSTALL_INDEX_URL = if ($env:INSTALL_INDEX_URL) { $env:INSTALL_INDEX_URL } else { "" }
$INSTALL_SKIP_POST_CHECK = if ($env:INSTALL_SKIP_POST_CHECK) { $env:INSTALL_SKIP_POST_CHECK } else { 0 }

$global:ACTION = "Install"

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
    $argList = @()
    for ($i = 0; $i -lt $RemainingArgs.Count; $i++) {
        $value = $RemainingArgs[$i + 1]
        switch ($RemainingArgs[$i]) {
            "--system-reserved" {
                $escapedJsonString = $value -replace '"', '\`"'
                $envList += "GPUSTACK_SYSTEM_RESERVED=`"$escapedJsonString`""
                $i++
            }
            default {
                $argList += $RemainingArgs[$i]
            }
        }
    }

    $envList += "APPDATA=$env:APPDATA"
    $envListString = $envList -join " "
    $argListString = $argList -join " "

    return $argListString, $envListString
}

# Get value of a script argument by name. Return "" if not found.
function Get-Arg-Value {
    param (
        [string]$ArgName,
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$ScriptArgs
    )

    for ($i = 0; $i -lt $ScriptArgs.Length; $i++) {
        $arg = $ScriptArgs[$i]

        # Handle equal sign passed arguments
        if ($arg -like "--$ArgName=*" -or $arg -like "-$ArgName=*") {
            return $arg.Split('=', 2)[1]
        }

        # Handle space passed arguments
        if ($arg -eq "--$ArgName" -or $arg -eq "-$ArgName") {
            if ($i + 1 -lt $ScriptArgs.Length) {
                return $ScriptArgs[$i + 1]
            }
        }
    }

    return ""
}

# Function to check if a port is available
function Check-PortAvailability {
    param([int]$Port)

    $connection = Get-NetTCPConnection -LocalPort $Port -LocalAddress '0.0.0.0' -State 'Listen' -ErrorAction SilentlyContinue

    if ($connection) {
        return $false  # Port is in use
    }
    else {
        return $true   # Port is available
    }
}

# Function to check if the server and worker ports are available
function Check-Port {
    param (
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$ScriptArgs
    )
    if (Get-Command gpustack -ErrorAction SilentlyContinue) {
        # skip on upgrade
        return
    }

    $configFile = Get-Arg-Value -ArgName "config-file" @ScriptArgs
    if ($configFile) {
        return
    }

    $serverPort = Get-Arg-Value -ArgName "port" @ScriptArgs
    $workerPort = Get-Arg-Value -ArgName "worker-port" @ScriptArgs
    $sslEnabled = Get-Arg-Value -ArgName "ssl-keyfile" @ScriptArgs

    if (-not $serverPort) {
        $serverPort = 80
        if ($sslEnabled) {
            $serverPort = 443
        }
    }

    if (-not $workerPort) {
        $workerPort = 10150
    }

    if (-not (Check-PortAvailability -Port $serverPort)) {
        throw "Server port $serverPort is already in use! Please specify a different port by using --port <YOUR_PORT>."
    }

    if (-not (Check-PortAvailability -Port $workerPort)) {
        throw "Worker port $workerPort is already in use! Please specify a different port by using --worker-port <YOUR_PORT>."
    }
}

# Function to print completion message.
function Print-Complete-Message {
    param (
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$ScriptArgs
    )
    $usageHint = ""
    $firewallHint = ""
    if ($ACTION -eq "Install") {
        $dataDir = Get-Arg-Value -ArgName "data-dir" @ScriptArgs
        if ([string]::IsNullOrEmpty($dataDir)) {
            $dataDir = "$env:APPDATA\gpustack"
        }

        $configFile = Get-Arg-Value -ArgName "config-file" @ScriptArgs
        $serverUrl = Get-Arg-Value -ArgName "server-url" @ScriptArgs
        if ([string]::IsNullOrEmpty($serverUrl)) {
            $serverUrl = Get-Arg-Value -ArgName "s" @ScriptArgs # try short form
        }

        # Skip printing the usage hint for workers and advanced users using config file.
        if ([string]::IsNullOrEmpty($serverUrl) -and [string]::IsNullOrEmpty($configFile)) {
            $serverUrl = "localhost"
            $serverHost = Get-Arg-Value -ArgName "host" @ScriptArgs
            if (-not [string]::IsNullOrEmpty($serverHost)) {
                $serverUrl = "${serverHost}"
            }
            $serverPort = Get-Arg-Value -ArgName "port" @ScriptArgs
            if (-not [string]::IsNullOrEmpty($serverPort)) {
                $serverUrl = "${serverUrl}:${serverPort}"
            }

            $sslEnabled = Get-Arg-Value -ArgName "ssl-keyfile" @ScriptArgs
            if (-not [string]::IsNullOrEmpty($sslEnabled)) {
                $serverUrl = "https://${serverUrl}"
            }
            else {
                $serverUrl = "http://${serverUrl}"
            }

            $passwordHint = ""
            $bootstrapPassword = Get-Arg-Value -ArgName "bootstrap-password" @ScriptArgs
            if ([string]::IsNullOrEmpty($bootstrapPassword)) {
                $passwordHint = "To get the default password, run 'Get-Content -Path `"${dataDir}\initial_admin_password`" -Raw'.`n"
            }

            $usageHint = "`n`nGPUStack UI is available at ${serverUrl}.`nDefault username is 'admin'.`n${passwordHint}`n"
        }
        $firewallHint = "Note: The Windows firewall may be enabled, and you may need to add rules to allow access."
    }


    Write-Host "$ACTION complete. ${usageHint}${firewallHint}"
}

function Refresh-ChocolateyProfile {
    $chocoInstallPath = [System.Environment]::GetEnvironmentVariable("ChocolateyInstall", [System.EnvironmentVariableTarget]::User)

    if (-not $chocoInstallPath) {
        $chocoInstallPath = [System.Environment]::GetEnvironmentVariable("ChocolateyInstall", [System.EnvironmentVariableTarget]::Machine)
    }

    if (Get-Command choco -ErrorAction SilentlyContinue) {
        $chocoPath = (Get-Command choco).Path
        $chocoInstallPath = Split-Path -Path (Split-Path -Path $chocoPath -Parent) -Parent
    }

    if (-not (Test-Path -Path $chocoInstallPath)) {
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
    $CURRENT_VERSION = $null
    if (Get-Command python -ErrorAction SilentlyContinue) {
        $PYTHON_VERSION = python -c 'import sys; print(sys.version_info.major * 10 + sys.version_info.minor)'
        $CURRENT_VERSION = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
        $pythonSource = $(Get-Command python).Source
        $isDirty = (($null -eq $PYTHON_VERSION) -or ($PYTHON_VERSION -eq "")) -and ($pythonSource -match "WindowsApps")

        if ($isDirty) {
            Log-Info "Python command is just alias for open Windows Store, clean it up..."
            Remove-Item "$env:LOCALAPPDATA\Microsoft\WindowsApps\python.exe" -ErrorAction SilentlyContinue
            Remove-Item "$env:LOCALAPPDATA\Microsoft\WindowsApps\python3.exe" -ErrorAction SilentlyContinue
        }
        elseif ($PYTHON_VERSION -lt 40 -or $PYTHON_VERSION -ge 43) {
            throw "Python version $CURRENT_VERSION is not supported. Please use Python 3.10, 3.11, or 3.12."
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

            Log-Info "Pipx installed successfully."
        }
        catch {
            throw "Failed to install Pipx: `"$($_.Exception.Message)`""
        }
    }
    else {
        Log-Info "Pipx already installed."
    }

    pipx ensurepath --force
    if ($LASTEXITCODE -ne 0) {
        throw "failed to run pipx ensurepath."
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
    if (Get-Command gpustack -ErrorAction SilentlyContinue) {
        $global:ACTION = "Upgrade"
        Log-Info "GPUStack already installed, Upgrading..."
    }

    try {
        Log-Info "$ACTION GPUStack..."
        $installArgs = @()
        if ($INSTALL_INDEX_URL) {
            $installArgs += "--index-url=$INSTALL_INDEX_URL"
        }

        Log-Info "$ACTION GPUStack with $($installArgs -join ' ') $INSTALL_PACKAGE_SPEC"

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
            $executableValue = ""
            foreach ($line in $configContent) {
                if ($line.StartsWith("home =")) {
                    $homeValue = $line.Split("=")[1].Trim()
                }

                if ($line.StartsWith("executable =")) {
                    $executableValue = $line.Split("=")[1].Trim()
                }
            }

            if ([string]::IsNullOrEmpty($executableValue)) {
                $executableValue = Join-Path $homeValue "python.exe"
            }

            if (-not (Test-Path -Path $homeValue) -or -not (Test-Path -Path $executableValue)) {
                Log-Warn "Current pipx config is invalid with non-existent paths: home path $homeValue or executable path $executableValue. Trying to refresh shared environment."
                python -m venv --clear $pipxSharedEnv
                if ($LASTEXITCODE -ne 0) {
                    throw "failed to refresh virtual environment."
                }
            }
        }

        Log-Info "$ACTION GPUStack with pipx and python $pythonPath..."
        if ($ACTION -ieq "Upgrade") {
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

        # Workaround for issue #581
        pipx inject gpustack pydantic==2.9.2 --force
        if ($LASTEXITCODE -ne 0) {
            throw "failed to run pipx inject."
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

        Log-Info "$ACTION GPUStack successfully."
    }
    catch {
        throw "Failed to $ACTION GPUStack: `"$($_.Exception.Message)`""
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
                Log-Info "Stopped existing ${serviceName} successfully"
            }
            else {
                Log-Warn "Failed to stop existing ${serviceName} service: `"$($result)`""
            }

            $result = nssm remove $serviceName confirm
            if ($LASTEXITCODE -eq 0) {
                Log-Info "Removed existing ${serviceName} successfully"
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
        [string]$argListString,
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
        $gpustackEnvPath = Join-Path -Path $gpustackDirectoryPath -ChildPath "gpustack.env"

        $null = New-Item -Path $gpustackDirectoryPath -ItemType "Directory" -ErrorAction SilentlyContinue -Force
        $null = New-Item -Path $gpustackLogDirectoryPath -ItemType "Directory" -ErrorAction SilentlyContinue -Force

        # Load additional environment variables from gpustack.env file.
        $additionalEnvVars = @()
        if (Test-Path $gpustackEnvPath) {
            Log-Info "Loading environment variables from $gpustackEnvPath..."
            $envFileContent = Get-Content -Path $gpustackEnvPath -ErrorAction Stop
            foreach ($line in $envFileContent) {
                if ($line -match '^\s*#' -or $line -match '^\s*$') { continue }  # Skip comments and empty lines.
                $additionalEnvVars += $line.Trim()
            }
        }

        # Merge additional environment variables with the existing ones, separated by space.
        $finalEnvList = @($envListString) + $additionalEnvVars -join " "

        $null = nssm install $serviceName $exePath
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to install service $serviceName"
        }

        $appParams = "start $argListString"

        $commands = @(
            "nssm set $serviceName AppDirectory $gpustackDirectoryPath",
            "nssm set $serviceName AppParameters $appParams",
            "nssm set $serviceName DisplayName $serviceDisplayName",
            "nssm set $serviceName Description 'GPUStack aims to get you started with managing GPU devices, running LLMs and performing inference in a simple yet scalable manner.'",
            "nssm set $serviceName Start SERVICE_AUTO_START",
            "nssm set $serviceName ObjectName LocalSystem",
            "nssm set $serviceName AppExit Default Restart",
            "nssm set $serviceName AppStdout $gpustackLogPath",
            "nssm set $serviceName AppStderr $gpustackLogPath",
            "nssm set $serviceName AppEnvironmentExtra $finalEnvList"
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

function Check-GPUStackService {
    param (
        [int]$Retries = 3,
        [int]$Interval = 2
    )

    if ($INSTALL_SKIP_POST_CHECK -eq 1) {
        return
    }

    $serviceName = "GPUStack"
    $appDataPath = $env:APPDATA
    $gpustackDirectoryName = "gpustack"
    $gpustackLogPath = Join-Path -Path (Join-Path -Path $appDataPath -ChildPath $gpustackDirectoryName) -ChildPath "log/gpustack.log"

    Log-Info "Waiting for the service to initialize..."
    Start-Sleep -s 10

    for ($i = 1; $i -le $Retries; $i++) {
        $status = nssm status $serviceName
        if ($status -eq 'SERVICE_RUNNING') {
            # Check abnormal exit from  nssm event logs
            $events = Get-EventLog -LogName Application -Source nssm -Newest 20 | Where-Object { $_.TimeGenerated -ge (Get-Date).AddSeconds(-30) }
            $hasError = $false

            foreach ($appEvent in $appEvents) {
                if ($appEvent.Message -match "$serviceName" -and $appEvent.Message -match "exit code") {
                    $hasError = $true
                    break
                }
            }

            if ($hasError) {
                throw "GPUStack service is running but exited abnormally. Please check logs at: $gpustackLogPath for details."
            }

            return
        }

        Log-Info "Service not ready, retrying in $Interval seconds ($i/$Retries)..."
        Start-Sleep -s $Interval
    }

    throw "GPUStack service failed to start. Please check the logs at: $gpustackLogPath for details."
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
                    Log-Info "Stopped ${serviceName} successfully"
                }
                else {
                    Log-Warn "Failed to stop existing ${serviceName} service: `"$($result)`""
                }

                $result = nssm remove $serviceName confirm
                if ($LASTEXITCODE -eq 0) {
                    Log-Info "Removed ${serviceName} successfully"
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
            Log-Info "Uninstalled package ${packageName} successfully."
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
    Check-Port @args
    Install-Chocolatey
    Install-Python
    Install-NSSM
    Install-GPUStack
    Create-UninstallScript
    $argResult = Get-Arg @args
    Setup-GPUStackService -argListString $argResult[0] -envListString $argResult[1]
    Check-GPUStackService
    Print-Complete-Message @args
}
catch {
    Log-Fatal "Failed to install GPUStack: `"$($_.Exception.Message)`""
}
