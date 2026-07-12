# Whetstone Inspector file contract

The Inspector is a stateless adapter for teams that already have item-level eval
outcomes. It does not require a Whetstone-native examiner bank.

## Exam

JSON array or JSONL. `item_id` is required; `domain` and content fields are
optional.

```json
{"item_id":"support-17","domain":"support","prompt":"Disposable example"}
```

## Exposure

JSON array or JSONL. Exact identity can be declared through `item_id`, `id`,
`exposure_key`, `content_hash`, or exact equality of a scalar `prompt`, `content`,
`input`, `task`, or `question` field. A `source` or `path` is reported without
echoing the training content.

```json
{"item_id":"support-17","source":"training.jsonl:412"}
```

## Results

Either an object keyed by item ID or JSONL rows. Baseline and candidate must
cover the identical complete post-quarantine cohort.

```json
{"item_id":"support-18","passed":true}
```

## Decision

The default strict policy requires at least one gain, zero regressions, and an
exact two-sided McNemar p-value at or below 0.05. A higher aggregate score does
not erase a regression. Incomplete cohorts HOLD rather than being silently
intersected.
