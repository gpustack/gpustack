<#
.SYNOPSIS
    A script to run GPUStack server or worker.

.DESCRIPTION
    Run GPUStack server or worker with the specified settings.

.PARAMETER ConfigFile
    Path to the YAML config file.
    Alias: config-file

.PARAMETER DataDir
    Common settings. Directory to store data. Default is OS specific.
    Alias: data-dir

.PARAMETER Token
    Common settings. Shared secret used to add a worker.
    Alias: token

.PARAMETER ServerHost
    Server settings. Host to bind the server to.
    Alias: server-host

.PARAMETER ServerPort
    Server settings. Port to bind the server to.
    Alias: server-port

.PARAMETER DatabaseURL
    Server settings. URL of the database. Example: postgresql://user:password@hostname:port/db_name.
    Alias: database-url

.PARAMETER DisableWorker
    Server settings. Disable embedded worker.
    Alias: disable-worker

.PARAMETER BootstrapPassword
    Server settings. Initial password for the default admin user. Random by default.
    Alias: bootstrap-password

.PARAMETER SystemReservedMemory
    Server settings. The system reserves resources for the worker during scheduling, measured in GiB.
    By default, 1 GiB of memory is reserved.
    Alias: system-reserved-memory

.PARAMETER SystemReservedGPUMemory
    Server settings. The system reserves resources for the worker during scheduling, measured in GiB.
    By default, 1 GiB of GPU memory is reserved.
    Alias: system-reserved-gpu-memory

.PARAMETER SSLKeyFile
    Server settings. Path to the SSL key file.
    Alias: ssl-key-file

.PARAMETER SSLCertFile
    Server settings. Path to the SSL certificate file.
    Alias: ssl-cert-file

.PARAMETER ForceAuthLocalhost
    Server settings. Force authentication for requests originating from localhost (127.0.0.1).
    When set to True, all requests from localhost will require authentication.
    Alias: force-auth-localhost


.PARAMETER ServerURL
    Worker settings. Server to connect to.
    Alias: server-url

.PARAMETER WorkerIP
    Worker settings. IP address of the worker node. Auto-detected by default.
    Alias: worker-ip

.PARAMETER EnableMetrics
    Worker settings. IP address of the worker node. Auto-detected by default.
    Alias: enable-metrics

.PARAMETER MetricsPort
    Worker settings. Enable metrics.
    Alias: metrics-port

.PARAMETER WorkerPort
    Worker settings. Port to expose metrics.
    Alias: worker-port

.PARAMETER LogDir
    Worker settings. Directory to store logs.
    Alias: log-dir

.EXAMPLE
    .\install.ps1

    You can start the GPUStack server by running the command on a server node.

.EXAMPLE
    .\install.ps1 -server-url http://myserver -token mytoken

    You can add additional workers to form a GPUStack cluster by running the command on worker nodes.
#>

[CmdletBinding()]
param (
    [Alias("config-file")]
    [Parameter()]
    [String]$ConfigFile,

    [Alias("data-dir")]
    [Parameter()]
    [String]$DataDir,

    [Parameter()]
    [String]$Token,

    [Alias("server-host")]
    [Parameter()]
    [String]$ServerHost = "0.0.0.0",

    [Alias("server-port")]
    [Parameter()]
    [String]$ServerPort,

    [Alias("database-url")]
    [Parameter()]
    [String]$DatabaseURL,

    [Alias("bootstrap-password")]
    [Parameter()]
    [String]$BootstrapPassword,

    [Alias("disable-worker")]
    [Parameter()]
    [Switch]$DisableWorker,

    [Alias("system-reserved-memory")]
    [Parameter()]
    [int]$SystemReservedMemory = 1,

    [Alias("system-reserved-gpu-memory")]
    [Parameter()]
    [int]$SystemReservedGPUMemory = 1,

    [Alias("ssl-key-file")]
    [Parameter()]
    [String]$SSLKeyFile,

    [Alias("ssl-cert-file")]
    [Parameter()]
    [String]$SSLCertFile,

    [Alias("force-auth-localhost")]
    [Parameter()]
    [switch]$ForceAuthLocalhost,

    [Alias("server-url")]
    [Parameter()]
    [String]$ServerURL,

    [Alias("worker-ip")]
    [Parameter()]
    [String]$WorkerIP,

    [Alias("enable-metrics")]
    [Parameter()]
    [Bool]$EnableMetrics = $true,

    [Alias("metric-port")]
    [Parameter()]
    [Int]$MetricsPort = 10151,

    [Alias("worker-port")]
    [Parameter()]
    [Int]$WorkerPort = 10150,

    [Alias("log-dir")]
    [Parameter()]
    [String]$LogDir,

    [Switch]$Help
)

if ($Help) {
    Get-Help -Full $MyInvocation.MyCommand.Definition
    exit
}

$ErrorActionPreference = "Stop"

$INSTALL_PACKAGE_SPEC = if ($env:INSTALL_PACKAGE_SPEC) { $env:INSTALL_PACKAGE_SPEC } else { "gpustack" }
$INSTALL_PRE_RELEASE = if ($env:INSTALL_PRE_RELEASE) { $env:INSTALL_PRE_RELEASE } else { 0 }

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

    if ($ConfigFile) {
        $envList += "GPUSTACK_CONFIG_File=$ConfigFile"
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

    if ($SystemReservedMemory -or $SystemReservedGPUMemory) {
        $reserved = @{
            memory     = $SystemReservedMemory
            gpu_memory = $SystemReservedGPUMemory
        }

        $jsonString = $reserved | ConvertTo-Json -Compress
        $escapedJsonString = $jsonString -replace '"', '\`"'
        $envList += "GPUSTACK_SYSTEM_RESERVED=`"$escapedJsonString`""
    }

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

    if ($EnableMetrics -eq $true -or $EnableMetrics -eq 1) {
        $envList += "GPUSTACK_ENABLE_METRICS=true"
    } else {
        $envList += "GPUSTACK_ENABLE_METRICS=false"
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
    if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
        try {
            Log-Info "Installing Python..."
            $null = choco install python --version=3.10.11 -y
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
        Log-Fatal "Python version is $PYTHON_VERSION, which is less than 3.10. Please upgrade Python to at least version 3.10."
    }

    if (-not (Get-Command pipx -ErrorAction SilentlyContinue)) {
        try {
            Log-Info "Pipx could not be found. Attempting to install..."

            python -m pip install pipx
            if ($LASTEXITCODE -ne 0) {
                Log-Fatal "failed to install pipx."
            }

            pipx ensurepath
            if ($LASTEXITCODE -ne 0) {
                Log-Fatal "failed to run pipx ensurepath."
            }

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
        if ($LASTEXITCODE -ne 0) {
            Log-Fatal "failed to install nssm."
        }

        Refresh-ChocolateyProfile
        if ($LASTEXITCODE -ne 0) {
            Log-Fatal "failed to refresh chocolatey profile."
        }

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
        if ($LASTEXITCODE -ne 0) {
            Log-Fatal "failed to install $INSTALL_PACKAGE_SPEC."
        }

        pipx ensurepath
        if ($LASTEXITCODE -ne 0) {
            Log-Fatal "failed to run pipx ensurepath."
        }

        Log-Info "Updating PATH environment variable..."

        $pipEnv = (pipx environment --value PIPX_BIN_DIR)
        if ($LASTEXITCODE -ne 0) {
            Log-Fatal "failed to run pipx environment."
        }

        $env:Path = "$pipEnv;$env:Path"

        $currentPath = [Environment]::GetEnvironmentVariable("PATH", "Machine")
        if (!$currentPath.Contains($pipEnv)) {
            [Environment]::SetEnvironmentVariable("Path", $env:Path, [EnvironmentVariableTarget]::Machine)
        } else {
            Log-Info "Path already contains $pipEnv"
        }
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

            $result = nssm remove $serviceName confirm
            Log-Info "Removed ${serviceName} result: $result"
        }
        catch {
            Log-Fatal "Failed to stop and remove existing ${serviceName} service: `"$($_.Exception.Message)`""
        }
    }

    try {
        Log-Info "Creating ${serviceName} service..."

        $appDataPath = $env:APPDATA
        $gpustackDirectoryName = "gpustack"
        $gpustackDirectoryPath = Join-Path -Path $appDataPath -ChildPath $gpustackDirectoryName

        $gpustackLogDirectoryPath = Join-Path -Path $gpustackDirectoryPath -ChildPath "log"
        $gpustackLogPath = Join-Path -Path $gpustackLogDirectoryPath -ChildPath "gpustack.log"

        $null = New-Item -Path $gpustackDirectoryPath -ItemType "Directory" -ErrorAction SilentlyContinue -Force
        $null = New-Item -Path $gpustackLogDirectoryPath -ItemType "Directory" -ErrorAction SilentlyContinue -Force

        $envListString = Get-Arg

        $null = nssm install $serviceName $exePath
        if ($LASTEXITCODE -ne 0) {
            Log-Fatal "Failed to install service $serviceName"
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
            if ($LASTEXITCODE -ne 0) { Log-Fatal "Failed to run nssm set environment: $cmd" }
        }

        Log-Info "Starting ${serviceName} service..."
        $null = nssm start $serviceName -y
        if ($LASTEXITCODE -ne 0) {
            Log-Fatal "Failed to start service $serviceName"
        }

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
                Log-Fatal "Failed to dump service $serviceName"
            }
        }
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
                Log-Info "Stopping existing ${serviceName} service..."
                $result = nssm stop $serviceName confirm
                Log-Info "Stopped ${serviceName} result: $result"

                $result =nssm remove $serviceName confirm
                Log-Info "Removed ${serviceName} result: $result"
            }
            catch {
                Log-Fatal "Failed to stop and remove existing ${serviceName} service: `"$($_.Exception.Message)`""
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

        Log-Info "Cleaning up..."
        $appDataPath = $env:APPDATA
        $gpustackDirectoryPath = Join-Path -Path $appDataPath -ChildPath $packageName
        if (Test-Path -Path $gpustackDirectoryPath) {
            Get-ChildItem -Path $appDataPath -Filter $packageName | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
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
    Setup-GPUStackService
    Create-UninstallScript
}
catch {
    Log-Fatal "Failed to install GPUStack: `"$($_.Exception.Message)`""
}
