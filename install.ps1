<#
.SYNOPSIS
    A script to run GPUStack server or worker.

.DESCRIPTION
    Run GPUStack server or worker with the specified settings.

.PARAMETER DataDir
    Common settings. Directory to store data. Default is OS specific.

.PARAMETER Token
    Common settings. Shared secret used to add a worker.

.PARAMETER ServerHost
    Server settings. Host to bind the server to.

.PARAMETER ServerPort
    Server settings. Port to bind the server to.

.PARAMETER DatabaseURL
    Server settings. URL of the database. Example: postgresql://user:password@hostname:port/db_name.

.PARAMETER DisableWorker
    Server settings. Disable embedded worker.

.PARAMETER BootstrapPassword
    Server settings. Initial password for the default admin user. Random by default.

.PARAMETER SystemReservedMemory
Server settings. The system reserves resources for the worker during scheduling, measured in GiB. \
        By default, 1 GiB of memory is reserved.

.PARAMETER SystemReservedGPUMemory
    Server settings. The system reserves resources for the worker during scheduling, measured in GiB. \
        By default, 1 GiB of GPU memory is reserved.

.PARAMETER SSLKeyFile
    Server settings. Path to the SSL key file.

.PARAMETER SSLCertFile
    Server settings. Path to the SSL certificate file.

.PARAMETER ForceAuthLocalhost
    Server settings. Force authentication for requests originating from localhost (127.0.0.1)."
        "When set to True, all requests from localhost will require authentication.

.PARAMETER ServerURL
    Worker settings. Server to connect to.

.PARAMETER WorkerIP
    Worker settings. IP address of the worker node. Auto-detected by default.

.PARAMETER EnableMetrics
    Worker settings. IP address of the worker node. Auto-detected by default.

.PARAMETER MetricsPort
    Worker settings. Enable metrics.

.PARAMETER WorkerPort
    Worker settings. Port to expose metrics.

.PARAMETER LogDir
    Worker settings. Port to bind the worker to.

.PARAMETER InstallPackageSpec
    Install settings. The package specification to install. Default is "gpustack".

.PARAMETER InstallPreRelease
    Install settings. Install pre-release versions. Default is false.

.EXAMPLE
    .\install.ps1

    You can start the GPUStack server by running the command on a server node.

.EXAMPLE
    .\install.ps1 --ServerURL http://myserver --Token mytoken

    You can add additional workers to form a GPUStack cluster by running the command on worker nodes.
#>

[CmdletBinding()]

param (
    [Parameter()]
    [String]$DataDir,

    [Parameter()]
    [String]$Token,

    [Parameter()]
    [String]$ServerHost = "0.0.0.0",

    [Parameter()]
    [String]$ServerPort,

    [Parameter()]
    [String]$DatabaseURL,

    [Parameter()]
    [SecureString]$BootstrapPassword,

    [Parameter()]
    [Switch]$DisableWorker,

    [Parameter()]
    [int]$SystemReservedMemory = 1,

    [Parameter()]
    [int]$SystemReservedGPUMemory = 1,

    [Parameter()]
    [String]$SSLKeyFile,

    [Parameter()]
    [String]$SSLCertFile,

    [Parameter()]
    [switch]$ForceAuthLocalhost,

    [Parameter()]
    [String]$ServerURL,

    [Parameter()]
    [String]$WorkerIP,

    [Parameter()]
    [Bool]$EnableMetrics = $true,

    [Parameter()]
    [Int]$MetricsPort = 10151,

    [Parameter()]
    [Int]$WorkerPort = 10150,

    [Parameter()]
    [String]$LogDir,

    [Parameter()]
    [String]$InstallPackageSpec,

    [Parameter()]
    [switch]$InstallPreRelease,


    [Switch]$Help
)

if ($Help) {
    Get-Help -Full $MyInvocation.MyCommand.Definition
    exit
}

$ErrorActionPreference = "Stop"

$INSTALL_PACKAGE_SPEC = if ($InstallPackageSpec) { $InstallPackageSpec } elseif ($env:INSTALL_PACKAGE_SPEC) { $env:INSTALL_PACKAGE_SPEC } else { "gpustack" }
$INSTALL_PRE_RELEASE = if ($InstallPreRelease) { 1 } elseif ($env:INSTALL_PRE_RELEASE) { $env:INSTALL_PRE_RELEASE } else { 0 }

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
    exit 1
}

# Function to check if the script is run as administrator
function Check-AdminPrivilege {
    if (-NOT ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
        Log-Fatal "This script must be run as Administrator. Please restart the script with Administrator privileges."
    }
}

# Function to detect the OS
function Check-OS {
    $OS = (Get-CimInstance -Class Win32_OperatingSystem).Caption
    if ($OS -notmatch "Windows") {
        Log-Fatal "Unsupported OS. Only Windows is supported."
    }
}

function Check-CUDA {
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        if (-not (Get-Command nvcc -ErrorAction SilentlyContinue)) {
            Log-Fatal "NVIDIA GPU detected but CUDA is not installed. Please install CUDA."
        }
    }
}

function Get-Arg {
    Log-Info "Getting arguments from flags..."

    $envList = @()
    if ($Debug) {
        $envList += "GPUSTACK_DEBUG=true"
    }

    if ($DataDir) {
        $envList += "GPUSTACK_DATA_DIR=$DataDir"
    }

    if ($Token) {
        $envList += "GPUSTACK_TOKEN=$Token"
    }

    if ($ServerHost) {
        $envList += "GPUSTACK_HOST=$ServerHOST"
    }

    if ($ServerPort) {
        $envList += "GPUSTACK_PORT=$ServerPort"
    }

    if ($DatabaseURL) {
        $envList += "GPUSTACK_DATABASE_URL=$DatabaseURL"
    }

    if ($BootstrapPassword) {
        $envList += "GPUSTACK_BOOTSTRAP_PASSWORD=$BootstrapPassword"
    }

    if ($DisableWorker) {
        $envList += "GPUSTACK_DISABLE_WORKER=true"
    }

    # if ($SystemReservedMemory -or $SystemReservedGPUMemory) {
    #     $reserved = @{
    #         memory     = $SystemReservedMemory
    #         gpu_memory = $SystemReservedGPUMemory
    #     }

    #     $jsonString = $reserved | ConvertTo-Json -Compress
    #     $escapedJsonString = $jsonString -replace '"', '\`"'
    #     $envList += "GPUSTACK_SYSTEM_RESERVED=`"$escapedJsonString`""
    # }

    if ($SSLKeyFile) {
        $envList += "GPUSTACK_SSL_KEY_FILE=$SSLKeyFile"
    }

    if ($SSLCertFile) {
        $envList += "GPUSTACK_SSL_CERT_FILE=$SSLCertFile"
    }

    if ($ForceAuthLocalhost) {
        $envList += "GPUSTACK_FORCE_AUTH_LOCALHOST= true"
    }

    if ($ServerURL) {
        $envList += "GPUSTACK_SERVER_URL=$ServerURL"
    }

    if ($WorkerIP) {
        $envList += "GPUSTACK_WORKER_IP=$WorkerIP"
    }

    if ($WorkerPort) {
        $envList += "GPUSTACK_WORKER_PORT=$WorkerPort"
    }

    if ($EnableMetrics) {
        $envList += "GPUSTACK_ENABLE_METRICS=true"
    }

    if ($MetricsPort) {
        $envList += "GPUSTACK_METRICS_PORT=$MetricsPort"
    }

    if ($LogDir) {
        $envList += "GPUSTACK_LOG_DIR=$LogDir"
    }

    $envList += "APPDATA=$env:APPDATA"

    $envListString = $envList -join " "

    return $envListString
}

function Refresh-ChocolateyProfile {
    $chocoInstallPath = [System.Environment]::GetEnvironmentVariable("ChocolateyInstall", "Machine")
    if (-not $chocoInstallPath) {
        Log-Fatal "Chocolatey installation path not found. Ensure Chocolatey is installed correctly."
    }

    $chocoHelpersPath = Join-Path -Path $chocoInstallPath -ChildPath "helpers\chocolateyProfile.psm1"
    try {
        Import-Module $chocoHelpersPath -ErrorAction Stop
        if (-not (Get-Command refreshenv -ErrorAction SilentlyContinue)) {
            Log-Fatal "Could not find 'refreshenv'. Something is wrong with Chocolatey installation."
        }

        refreshenv
    }
    catch {
        Log-Fatal "Failed to import Chocolatey profile. Ensure Chocolatey is installed correctly."
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
        Log-Fatal "Failed to install Chocolatey: `"$($_.Exception.Message)`""
    }
}

function Install-Python {
    if (-not (Get-Command py40on -ErrorAction SilentlyContinue)) {
        try {
            Log-Info "Installing Python..."
            choco install python --version=3.10.11 -y
            Refresh-ChocolateyProfile
            Log-Info "Python installed successfully."
        }
        catch {
            Log-Fatal "Failed to install Python: `"$($_.Exception.Message)`""
        }
    }
    else {
        Log-Info "Python already installed."
    }

    $PYTHON_VERSION = python -c "import sys; print(sys.version_info.major * 10 + sys.version_info.minor)"
    if ($PYTHON_VERSION -lt 40) {
        Log-Fatal "Python version is less than 3.10. Please upgrade Python to at least version 3.10."
    }

    if (-not (Get-Command pipx -ErrorAction SilentlyContinue)) {
        try {
            Log-Info "Pipx could not be found. Attempting to install..."
            python -m pip install pipx
            pipx ensurepath
            Log-Info "Pipx installed successfully."
        }
        catch {
            Log-Fatal "Failed to install Pipx: `"$($_.Exception.Message)`""
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
        Refresh-ChocolateyProfile
        Log-Info "NSSM installed successfully."
    }
    catch {
        Log-Fatal "Failed to install NSSM: `"$($_.Exception.Message)`""
    }
}

function Install-GPUStack {
    if (Get-Command gpustack -ErrorAction SilentlyContinue) {
        Log-Info "GPUStack already installed."
        return
    }

    try {
        Log-Info "Installing GPUStack..."
        $installArgs = ""
        if ($INSTALL_PRE_RELEASE -eq 1) {
            $installArgs = "--pip-args='--pre'"
        }

        Log-Info "Installing GPUStack with $INSTALL_PACKAGE_SPEC $installArgs"
        
        $pythonPath = Get-Command python | Select-Object -ExpandProperty Source
        $env:PIPX_DEFAULT_PYTHON = $pythonPath

        Log-Info "Installing GPUStack with pipx and pythin $pythonPath..."
        
        pipx install $installArgs $INSTALL_PACKAGE_SPEC --force --verbose
        pipx ensurepath

        Log-Info "Updating PATH environment variable..."
        
        $pipEnv = (pipx environment --value PIPX_BIN_DIR)
        $env:Path = "$pipEnv;$env:Path"

        [Environment]::SetEnvironmentVariable("Path", $env:Path, [EnvironmentVariableTarget]::Machine)
    }
    catch {
        Log-Fatal "Failed to install GPUStack: `"$($_.Exception.Message)`""
    }
}

function Setup-GPUStackService {
    $serviceName = "GPUStack"
    $serviceDisplayName = "GPUStack"
    $exePath = $(Get-Command gpustack).Source
    $exeFile = Get-Item -Path $exePath

    if ($exeFile.Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
        $exePath = (Get-Item -Path $exeFile).Target
    }

    # Check if the service already exists.
    $gpustack = Get-Service -Name $serviceName -ErrorAction SilentlyContinue
    if ($null -ne $gpustack) {
        try {
            Log-Info "Stopping existing ${serviceName} service, creatig a new one..."
            $result = nssm stop $serviceName confirm
            Log-Info "Stopped ${serviceName} result: $result"

            $result = sm remove $serviceName confirm
            Log-Info "Removed ${serviceName} result: $result"
        }
        catch {
            Log-Error "Failed to stop and remove existing ${serviceName} service: `"$($_.Exception.Message)`""
        }
    }

    try {
        Log-Info "Creating ${serviceName} service..."

        $appDataPath = $env:APPDATA
        $gpustackDirectoryName = "gpustack"
        $gpustackDirectoryPath = Join-Path -Path $appDataPath -ChildPath $gpustackDirectoryName

        $gpustackLogDirectoryPath = Join-Path -Path $gpustackDirectoryPath -ChildPath "log"
        $gpustackStdoutLogPath = Join-Path -Path $gpustackLogDirectoryPath -ChildPath "gpustack_stdout.log"
        $gpustackStderrLogPath = Join-Path -Path $gpustackLogDirectoryPath -ChildPath "gpustack_stderr.log"
        $envListString = Get-Arg

        $null = nssm install $serviceName $exePath
        $null = nssm set $serviceName AppDirectory $gpustackDirectoryPath
        $null = nssm set $serviceName AppParameters "start"

        $null = nssm set $serviceName DisplayName $serviceDisplayName
        $null = nssm set $serviceName Description "GPUStack aims to get you started with managing GPU devices, running LLMs and performing inference in a simple yet scalable manner."
        $null = nssm set $serviceName Start SERVICE_AUTO_START

        $null = nssm set $serviceName ObjectName LocalSystem
        $null = nssm set $serviceName AppExit Default Restart
        $null = nssm set $serviceName AppStdout $gpustackStdoutLogPath
        $null = nssm set $serviceName AppStderr $gpustackStderrLogPath

        $nssmSetEnvCommand = "nssm set $serviceName AppEnvironmentExtra $envListString"
        $null = Invoke-Expression $nssmSetEnvCommand

        Log-Info "Starting ${serviceName} service..."
        $null = nssm start $serviceName -y

        # Wait for the service to start for 60 seconds.
        # $startTime = Get-Date
        # while ((nssm status $serviceName) -ne 'SERVICE_RUNNING' -and ((Get-Date) - $startTime).TotalSeconds -lt 60) {
        #     Log-Info "Waiting for $serviceName service to start."
        #     Start-Sleep -s 5
        # }

        # Log-Info "${serviceName} service created and started successfully."
    }
    catch {
        Log-Fatal "Failed to setup ${serviceName}: `"$($_.Exception.Message)`""
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
    exit 1
}

# Function to check if the script is run as administrator
function Check-AdminPrivilege {
    if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
        Log-Fatal "This script must be run as Administrator. Please restart the script with Administrator privileges."
    }
}

# Function to detect the OS
function Check-OS {
    $OS = (Get-CimInstance -Class Win32_OperatingSystem).Caption
    if ($OS -notmatch "Windows") {
        Log-Fatal "Unsupported OS. Only Windows is supported."
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
                Log-Info "Stopping existing ${serviceName} service, creatig a new one..."
                $result = nssm stop $serviceName confirm
                Log-Info "Stopped ${serviceName} result: $result"

                $result =nssm remove $serviceName confirm
                Log-Info "Removed ${serviceName} result: $result"
            }
            catch {
                Log-Error "Failed to stop and remove existing ${serviceName} service: `"$($_.Exception.Message)`""
            }
        }else{
            Log-Info "No existing ${serviceName} service found."
        }


        if (-not(Get-Command pipx -ErrorAction SilentlyContinue)) {
            Log-Fatal "Pipx not found."
            return
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
    }
    catch {
        Log-Fatal "Failed to uninstall GPUStack: `"$($_.Exception.Message)`""
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
        $null = New-Item -ItemType Directory -Path $gpustackDirectoryPath -ErrorAction SilentlyContinue

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
    Setup-GPUStackService
    Create-UninstallScript
}
catch {
    Log-Fatal "Failed to install GPUStack: `"$($_.Exception.Message)`""
}
