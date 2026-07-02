$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
    $env:PYTHONPATH = "src"
    python -m pytest
    python -m bcv.usefulness
    python -m bcv.corpus_benchmark
    python -m bcv.recall_benchmark
    python -m bcv.taste
    python -m bcv.discovery --max-n 6
    python -m bcv.graph_agent --max-n 6 --proposal-file "sample_docs/graph_proposals.json"
    python -m bcv.research_foundry --mode scripted --rounds 2 --max-n 6
    python -m bcv.markdown_agent `
        --input "sample_docs/vendor_agreement.md" `
        --output ".bcv_runs/useful/vendor_agreement_edited.md" `
        --instruction "In the Scope section, add that Northstar Labs will provide a weekly deployment summary. Do not change payment terms, dates, parties, invoice IDs, headings, or citations." `
        --run-root ".bcv_runs/useful/markdown_agent"
    python -m bcv.experiments
    python -m bcv.lora_smoke
}
finally {
    Pop-Location
}
