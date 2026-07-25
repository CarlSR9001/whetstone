[CmdletBinding()]
param(
    [switch]$SkipFullTests
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot

function Assert-NativeSuccess {
    param([Parameter(Mandatory)][string]$Command)

    if ($LASTEXITCODE -ne 0) {
        throw "$Command failed with exit code $LASTEXITCODE"
    }
}

Push-Location $repo
try {
    $workingTreeStatus = @(git status --porcelain=v1 --untracked-files=all)
    Assert-NativeSuccess "git status"
    if ($workingTreeStatus.Count -gt 0) {
        throw "release checks require a clean working tree"
    }

    git diff --check
    Assert-NativeSuccess "git diff --check"
    python -m compileall -q src tests
    Assert-NativeSuccess "python compileall"
    node --check src/bcv/toolbox_static/app.js
    Assert-NativeSuccess "node --check"
    bash -n deploy/install-release.sh deploy/publish.sh
    Assert-NativeSuccess "bash -n"
    & "$PSScriptRoot/audit_publication.ps1" -IncludeUntracked
    Assert-NativeSuccess "publication audit"

    if (-not $SkipFullTests) {
        python -m pytest -q -p no:cacheprovider
        Assert-NativeSuccess "python pytest"
    }

    $dist = Join-Path $repo "dist"
    $dist = [System.IO.Path]::GetFullPath($dist)
    $repoPrefix = [System.IO.Path]::GetFullPath($repo).TrimEnd(
        [System.IO.Path]::DirectorySeparatorChar,
        [System.IO.Path]::AltDirectorySeparatorChar
    ) + [System.IO.Path]::DirectorySeparatorChar
    if (-not $dist.StartsWith($repoPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "refusing to clean dist outside repository: $dist"
    }
    if (Test-Path -LiteralPath $dist) {
        Remove-Item -LiteralPath $dist -Recurse -Force
    }
    python -m build
    Assert-NativeSuccess "python build"
    $wheel = Get-ChildItem -LiteralPath $dist -Filter *.whl |
        Sort-Object LastWriteTimeUtc -Descending |
        Select-Object -First 1
    if (-not $wheel) {
        throw "wheel build produced no artifact"
    }
    python "$PSScriptRoot/clean_install_smoke.py" --wheel $wheel.FullName
    Assert-NativeSuccess "clean-install smoke"
    Write-Host "release_check: PASS"
}
finally {
    Pop-Location
}
