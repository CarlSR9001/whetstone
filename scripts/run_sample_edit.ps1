$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
    $env:PYTHONPATH = "src"
    python -m bcv.markdown_agent `
        --input "sample_docs/vendor_agreement.md" `
        --output ".bcv_runs/useful/vendor_agreement_edited.md" `
        --instruction "In the Scope section, add that Northstar Labs will provide a weekly deployment summary. Do not change payment terms, dates, parties, invoice IDs, headings, or citations." `
        --run-root ".bcv_runs/useful/markdown_agent"
}
finally {
    Pop-Location
}

