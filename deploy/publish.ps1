[CmdletBinding()]
param(
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._@:-]*$')]
    [string]$SshHost = "vps2",

    [ValidateScript({ -not $_.StartsWith("-") })]
    [string]$GitRef = "HEAD",

    [ValidateSet("Wsl", "Native")]
    [string]$SshTransport = "Wsl",

    [ValidatePattern('^[A-Za-z0-9._-]+$')]
    [string]$WslDistribution = "Ubuntu",

    [switch]$LocalPreflightOnly,

    [ValidateSet("auto", "active", "inactive")]
    [string]$RollbackForgeState = $(
        if ($env:WHETSTONE_EXPECT_FORGE_ACTIVE) {
            $env:WHETSTONE_EXPECT_FORGE_ACTIVE
        }
        else {
            "auto"
        }
    )
)

$ErrorActionPreference = "Stop"

function Invoke-NativeCapture {
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [Parameter(Mandatory)][string[]]$Arguments
    )

    $output = @(& $FilePath @Arguments)
    if ($LASTEXITCODE -ne 0) {
        throw "$FilePath failed with exit code $LASTEXITCODE"
    }
    return ($output -join [Environment]::NewLine).Trim()
}

function Invoke-NativeChecked {
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [Parameter(Mandatory)][string[]]$Arguments
    )

    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$FilePath failed with exit code $LASTEXITCODE"
    }
}

function Invoke-TransportChecked {
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [Parameter(Mandatory)][string[]]$Arguments
    )

    if ($SshTransport -eq "Wsl") {
        Invoke-NativeChecked "wsl.exe" (@("-d", $WslDistribution, "--", $FilePath) + $Arguments)
    }
    else {
        Invoke-NativeChecked $FilePath $Arguments
    }
}

$repo = Invoke-NativeCapture "git" @("rev-parse", "--show-toplevel")
$git = @("-c", "safe.directory=$repo", "-C", $repo)
$branch = Invoke-NativeCapture "git" ($git + @("branch", "--show-current"))
$commit = Invoke-NativeCapture "git" ($git + @("rev-parse", "--verify", "$GitRef^{commit}"))

$workingTree = Invoke-NativeCapture "git" ($git + @("status", "--porcelain=v1", "--untracked-files=all"))
if ($workingTree) {
    throw "working tree is not clean"
}

if (-not $LocalPreflightOnly) {
    if ($branch -ne "main") {
        throw "releases must be cut from main (current branch: $branch)"
    }
    & git @git "merge-base" "--is-ancestor" $commit "main"
    if ($LASTEXITCODE -ne 0) {
        throw "target commit is not on main: $commit"
    }
    Invoke-NativeChecked "git" ($git + @("fetch", "--quiet", "origin", "main"))
    & git @git "merge-base" "--is-ancestor" $commit "origin/main"
    if ($LASTEXITCODE -ne 0) {
        throw "target commit is not present on public origin/main: $commit"
    }
}

$tempBase = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath())
$tempRoot = Join-Path $tempBase ("whetstone-publish-" + [guid]::NewGuid().ToString("N"))
$archive = Join-Path $tempRoot "whetstone-$commit.tar"
$extract = Join-Path $tempRoot "extract"
$remoteArchive = "/tmp/whetstone-$commit.tar"
$remoteInstaller = "/tmp/whetstone-install-$commit.sh"
$remoteUploaded = $false

try {
    New-Item -ItemType Directory -Path $extract | Out-Null
    Invoke-NativeChecked "git" ($git + @("archive", "--format=tar", "--output=$archive", $commit))
    Invoke-NativeChecked "tar" @("-xf", $archive, "-C", $extract)

    $versionFile = Join-Path $extract "src\bcv\_version.py"
    $provenanceScript = @'
import importlib.util
import os
import sys

os.environ.pop("WHETSTONE_BUILD_COMMIT", None)
spec = importlib.util.spec_from_file_location("_whetstone_release", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(module.build_commit())
'@
    $embeddedOutput = @($provenanceScript | & python - $versionFile)
    if ($LASTEXITCODE -ne 0) {
        throw "python archive provenance check failed with exit code $LASTEXITCODE"
    }
    $embeddedCommit = ($embeddedOutput -join [Environment]::NewLine).Trim()
    if ($embeddedCommit -ne $commit) {
        throw "local archive provenance '$embeddedCommit' does not match $commit"
    }

    $archiveSha256 = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
    Write-Host "commit=$commit"
    Write-Host "archive_sha256=$archiveSha256"
    if ($LocalPreflightOnly) {
        Write-Host "local_preflight=PASS"
        return
    }
    $transportArchive = $archive
    $transportInstaller = Join-Path $extract "deploy\install-release.sh"
    if ($SshTransport -eq "Wsl") {
        $transportArchive = Invoke-NativeCapture "wsl.exe" @("-d", $WslDistribution, "--", "wslpath", "-a", $archive)
        $transportInstaller = Invoke-NativeCapture "wsl.exe" @("-d", $WslDistribution, "--", "wslpath", "-a", $transportInstaller)
    }
    Invoke-TransportChecked "scp" @($transportArchive, "${SshHost}:$remoteArchive")
    $remoteUploaded = $true
    Invoke-TransportChecked "scp" @($transportInstaller, "${SshHost}:$remoteInstaller")
    Invoke-TransportChecked "ssh" @(
        $SshHost,
        "sudo", "bash", $remoteInstaller, $commit, $remoteArchive, $RollbackForgeState
    )
}
finally {
    if ($remoteUploaded) {
        try {
            Invoke-TransportChecked "ssh" @(
                $SshHost, "rm", "-f", "--", $remoteArchive, $remoteInstaller
            )
        }
        catch {
            Write-Warning "remote temporary-file cleanup failed: $($_.Exception.Message)"
        }
    }
    $resolvedTemp = [System.IO.Path]::GetFullPath($tempRoot)
    if (
        $resolvedTemp.StartsWith($tempBase, [System.StringComparison]::OrdinalIgnoreCase) -and
        (Split-Path -Leaf $resolvedTemp).StartsWith("whetstone-publish-") -and
        (Test-Path -LiteralPath $resolvedTemp)
    ) {
        Remove-Item -LiteralPath $resolvedTemp -Recurse -Force
    }
}
