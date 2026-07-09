"""The bank's metabolism: is the exam replenishing faster than it is consumed?

A private bank is not a static hidden folder; it is a resource with intake and
expenditure. Items are MINTED at the frontier, PROMOTED into service, and then
consumed — RETIRED when they saturate, BURNED when they are exposed outside
the trust boundary. If consumption outpaces minting, the instrument quietly
loses resolution until the gate can no longer certify anything (the resolution
statement in the gate report is the point-in-time symptom; this module is the
longitudinal cause).

Everything here is computed from the append-only bank_events.jsonl lifecycle
trail. No item contents are read or reported — only ids, domains, and events —
so a metabolism report is safe to share where the bank itself never goes.
"""

from __future__ import annotations

import html as html_lib
import json
from pathlib import Path
from typing import Any

CONSUMING_EVENTS = ("retired", "burned")


def load_events(root: str | Path) -> list[dict]:
    path = Path(root) / "bank_events.jsonl"
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def metabolism_report(root: str | Path) -> dict[str, Any]:
    events = load_events(root)
    totals: dict[str, int] = {}
    by_domain_minted: dict[str, int] = {}
    supply = 0
    supply_series: list[dict] = []
    for index, event in enumerate(events):
        kind = event["event"]
        totals[kind] = totals.get(kind, 0) + 1
        if kind == "minted":
            by_domain_minted[event.get("domain", "?")] = by_domain_minted.get(event.get("domain", "?"), 0) + 1
        if kind == "promoted":
            supply += 1
        elif kind == "retired" or (kind == "burned" and event.get("prior_status") == "promoted"):
            supply -= 1
        supply_series.append(
            {"n": index + 1, "event": kind, "supply": supply, "timestamp": event.get("timestamp", "")}
        )

    minted = totals.get("minted", 0)
    promoted = totals.get("promoted", 0)
    consumed = totals.get("retired", 0) + sum(
        1 for event in events if event["event"] == "burned" and event.get("prior_status") == "promoted"
    )
    if promoted == 0:
        sustainability = "no promotions recorded yet"
    elif consumed == 0:
        sustainability = "no consumption recorded yet; ratio undefined"
    elif promoted > consumed:
        sustainability = "replenishing: minting has outpaced consumption so far"
    elif promoted == consumed:
        sustainability = "break-even: every promoted item has been consumed"
    else:
        sustainability = "draining: consumption has outpaced promotion"

    return {
        "schema_version": 1,
        "root": str(root),
        "events_total": len(events),
        "totals": dict(sorted(totals.items())),
        "minted_by_domain": dict(sorted(by_domain_minted.items())),
        "current_promoted_supply": supply,
        "minted_total": minted,
        "promoted_total": promoted,
        "consumed_from_supply": consumed,
        # Minting measures intake; promotion measures usable replenishment.
        # Both matter because an enormous candidate queue cannot rescue a gate.
        "mint_to_consumption_ratio": round(minted / consumed, 3) if consumed else None,
        "promotion_to_consumption_ratio": round(promoted / consumed, 3) if consumed else None,
        "sustainability": sustainability,
        "supply_series": supply_series,
    }


def metabolism_summary(root: str | Path) -> dict[str, Any]:
    """Safe status payload: rates and totals, never item ids or the full series."""
    report = metabolism_report(root)
    return {
        key: report[key]
        for key in (
            "events_total",
            "totals",
            "minted_by_domain",
            "current_promoted_supply",
            "minted_total",
            "promoted_total",
            "consumed_from_supply",
            "mint_to_consumption_ratio",
            "promotion_to_consumption_ratio",
            "sustainability",
        )
    }


def _supply_svg(series: list[dict], width: int = 720, height: int = 160) -> str:
    if not series:
        return "<p>No lifecycle events yet.</p>"
    values = [point["supply"] for point in series]
    top = max(max(values), 1)
    bottom = min(min(values), 0)
    span = max(top - bottom, 1)
    pad = 8
    step = (width - 2 * pad) / max(len(values) - 1, 1)

    def x(index: int) -> float:
        return pad + index * step

    def y(value: int) -> float:
        return pad + (top - value) * (height - 2 * pad) / span

    points = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(values))
    zero_y = y(0)
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="promoted supply over lifecycle events">'
        f'<line x1="{pad}" y1="{zero_y:.1f}" x2="{width - pad}" y2="{zero_y:.1f}" stroke="#ccd3db"/>'
        f'<polyline points="{points}" fill="none" stroke="#33607f" stroke-width="2"/>'
        f'<circle cx="{x(len(values) - 1):.1f}" cy="{y(values[-1]):.1f}" r="4" fill="#33607f"/>'
        f"</svg>"
    )


def render_metabolism_html(report: dict[str, Any]) -> str:
    totals_rows = "".join(
        f"<tr><td>{html_lib.escape(kind)}</td><td>{count}</td></tr>"
        for kind, count in report["totals"].items()
    )
    domain_rows = "".join(
        f"<tr><td>{html_lib.escape(domain)}</td><td>{count}</td></tr>"
        for domain, count in report["minted_by_domain"].items()
    )
    ratio = report["mint_to_consumption_ratio"]
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Whetstone bank metabolism</title>
<style>body{{font:16px system-ui,sans-serif;max-width:900px;margin:48px auto;color:#18212b}}h1{{margin-bottom:4px}}.stat{{font-size:2rem;font-weight:800;color:#33607f}}table{{border-collapse:collapse;margin-top:12px}}th,td{{border:1px solid #ccd3db;padding:6px 12px;text-align:left}}th{{background:#f3f5f7}}svg{{width:100%;height:auto;margin-top:12px;border:1px solid #e2e7ea}}</style></head>
<body><h1>Whetstone bank metabolism</h1>
<p class="stat">{html_lib.escape(report["sustainability"])}</p>
<p>Current promoted supply: <strong>{report["current_promoted_supply"]}</strong> &middot;
consumed from supply: <strong>{report["consumed_from_supply"]}</strong> &middot;
mint-to-consumption ratio: <strong>{ratio if ratio is not None else "n/a"}</strong> &middot;
promotion-to-consumption ratio: <strong>{report["promotion_to_consumption_ratio"] if report["promotion_to_consumption_ratio"] is not None else "n/a"}</strong></p>
<h2>Promoted supply over lifecycle events</h2>
{_supply_svg(report["supply_series"])}
<h2>Event totals</h2><table><tr><th>event</th><th>count</th></tr>{totals_rows}</table>
<h2>Minted by domain</h2><table><tr><th>domain</th><th>count</th></tr>{domain_rows}</table>
<p>Computed from the append-only lifecycle trail; no item contents are read or shown.</p>
</body></html>"""


def write_metabolism_report(root: str | Path, output_dir: str | Path) -> tuple[Path, Path]:
    report = metabolism_report(root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "metabolism.json"
    html_path = output_dir / "metabolism.html"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    html_path.write_text(render_metabolism_html(report), encoding="utf-8")
    return json_path, html_path
