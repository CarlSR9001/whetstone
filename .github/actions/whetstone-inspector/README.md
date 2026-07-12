# Whetstone Inspector action

The action runs the same stateless file contract as `whetstone inspect`, uploads
the JSON receipt even when the gate blocks, and makes PASS/HOLD/BLOCK the job exit
code. It detects declared/exact exposure identity; it does not claim semantic
near-duplicate detection.

```yaml
jobs:
  promotion-gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: CarlSR9001/whetstone/.github/actions/whetstone-inspector@YOUR_REVIEWED_REF
        with:
          exam: eval/private_exam.jsonl
          exposure: eval/declared_training.jsonl
          baseline: eval/baseline_results.jsonl
          candidate: eval/candidate_results.jsonl
```

Do not commit a genuinely private exam just to use a hosted CI runner. Keep the
bank on a self-hosted runner or use Whetstone inside the customer trust boundary.
