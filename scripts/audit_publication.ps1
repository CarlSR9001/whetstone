[CmdletBinding()]
param([switch]$IncludeUntracked)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Push-Location $root
try {
    $files = if ($IncludeUntracked) { @(& git ls-files --cached --others --exclude-standard) } else { @(& git ls-files --cached) }
    $violations = [System.Collections.Generic.List[string]]::new()
    $forbiddenPaths = @(
        '(^|/)[.]bcv_runs(/|$)', '(^|/)gtp_logs(/|$)', '(^|/)tools(/|$)',
        '(^|/)results/private(/|$)', '(^|/)(adapters?|checkpoints?|model_cache)(/|$)', '[.]sqlite3$',
        '(^|/)(bank_state[.]json|private_promotion_exam[.]jsonl|grade_events[.]jsonl|bank_events[.]jsonl|[.]bank[.]lock)$',
        '[.]zip$'
    )
    $forbiddenContent = @(
        '(?i)C:\\Users\\[^\\\s]+',
        '(?i)(api[_-]?key|access[_-]?token|secret)\s*[:=]\s*["''][^"'']{8,}',
        '(?i)-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----',
        '(?i)sk-[A-Za-z0-9_-]{20,}'
    )
    $forbiddenResultContent = '(?i)"(prompt|raw_output|adapter_path|item_id)"\s*:'
    foreach ($relative in $files) {
        $normalized = $relative -replace '\\', '/'
        foreach ($pattern in $forbiddenPaths) {
            if ($normalized -match $pattern) { $violations.Add("forbidden path: $relative") }
        }
        $path = Join-Path $root $relative
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { continue }
        try { $content = Get-Content -LiteralPath $path -Raw -ErrorAction Stop } catch { continue }
        foreach ($pattern in $forbiddenContent) {
            if ($content -match $pattern) { $violations.Add("sensitive content pattern in: $relative") }
        }
        if ($normalized -match '^results/' -and $content -match $forbiddenResultContent) {
            $violations.Add("private result field in: $relative")
        }
    }
    Write-Host "Audited $($files.Count) publication candidate files."
    $remotes = @(& git remote)
    if ($remotes.Count -eq 0) { Write-Host 'Remote check: none configured (safe default).' }
    else { Write-Warning "Configured remotes: $($remotes -join ', ')" }
    if ($violations.Count -gt 0) {
        $violations | Sort-Object -Unique | ForEach-Object { Write-Host "FAIL: $_" -ForegroundColor Red }
        exit 1
    }
    Write-Host 'PASS: no configured publication guard pattern matched.'
    Write-Host 'Manual review is still required for every new result artifact.'
} finally { Pop-Location }
