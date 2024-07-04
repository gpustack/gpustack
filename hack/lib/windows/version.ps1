# Define the function to get version variables
function Get-GPUStackVersionVar {
    # Get the build date
    $BUILD_DATE = Get-Date -Format 'yyyy-MM-ddTHH:mm:ssZ'

    # Initialize other variables
    $GIT_TREE_STATE = "unknown"
    $GIT_COMMIT = "unknown"
    $GIT_VERSION = "unknown"

    # Check if the source was exported through git archive
    if ('%$Format:%' -eq '%') {
        $GIT_TREE_STATE = "archive"
        $GIT_COMMIT = '$Format:%H$'

        # Parse the version from '$Format:%D$'
        if ('%$Format:%D$' -match 'tag:\s+(v[^ ,]+)') {
            $GIT_VERSION = $matches[1]
        } else {
            $GIT_VERSION = $GIT_COMMIT.Substring(0, 7)
        }

        # Respect specified version
        $GIT_VERSION = $env:VERSION -or $GIT_VERSION
        return
    }

    # Return if git client is not found
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        $GIT_VERSION = $env:VERSION -or $GIT_VERSION
        return
    }

    # Find git info via git client
    $GIT_COMMIT = git rev-parse "HEAD^{commit}" 2>$null
    if ($LASTEXITCODE -eq 0) {
        # Check if the tree is clean or dirty
        $gitStatus = (git status --porcelain 2>$null)
        if ($gitStatus) {
            $GIT_TREE_STATE = "dirty"
        } else {
            $GIT_TREE_STATE = "clean"
        }

        # Get the version from HEAD
        $GIT_VERSION = git rev-parse --abbrev-ref HEAD 2>$null
        if ($LASTEXITCODE -eq 0) {
            # Check if HEAD is tagged
            $gitTag = git tag -l --contains HEAD 2>$null | Select-Object -First 1
            if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrEmpty($gitTag)) {
                $GIT_VERSION = $gitTag
            }
        }

        # Set version to '0.0.0' if the tree is dirty or version format does not match
        if ($GIT_TREE_STATE -eq "dirty" -or -not ($GIT_VERSION -match '^v([0-9]+)\.([0-9]+)(\.[0-9]+)?(-[0-9A-Za-z.-]+)?(\+[0-9A-Za-z.-]+)?$')) {
            $GIT_VERSION = "0.0.0"
        }

        # Respect specified version
        if ($env:VERSION) {
            $GIT_VERSION = $env:VERSION
        }
    }

    $global:BUILD_DATE = $BUILD_DATE
    $global:GIT_TREE_STATE = $GIT_TREE_STATE
    $global:GIT_COMMIT = $GIT_COMMIT
    $global:GIT_VERSION = $GIT_VERSION
}
