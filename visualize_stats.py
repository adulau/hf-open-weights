#!/usr/bin/env python3
"""Turn hf-open-weights aggregate statistics into a self-contained dashboard."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


DIMENSION_LABELS = {
    "organization": "Organizations",
    "country": "Countries",
    "language": "Languages",
    "license_class": "License classes",
    "license": "Licenses",
    "pipeline_tag": "Pipelines",
    "library_name": "Libraries",
    "training_info_status": "Training documentation",
    "gated": "Access",
}


def load_statistics(path: Path) -> dict[str, Any]:
    """Load and validate the subset of the statistics schema used by the view."""
    with path.open(encoding="utf-8") as source:
        data = json.load(source)
    if not isinstance(data, dict) or not isinstance(data.get("dimensions"), dict):
        raise ValueError("statistics JSON must contain a 'dimensions' object")
    for name in ("model_count", "downloads_total", "likes_total", "followers_total"):
        value = data.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"'{name}' must be a non-negative integer")
    for name, values in data["dimensions"].items():
        if not isinstance(values, dict) or any(
            not isinstance(label, str)
            or not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            for label, value in values.items()
        ):
            raise ValueError(f"dimension '{name}' must map labels to non-negative integers")
    return data


def render_dashboard(data: dict[str, Any], title: str, top: int) -> str:
    """Render a dependency-free HTML dashboard with embedded statistics."""
    if top < 1:
        raise ValueError("top must be at least 1")
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    labels = json.dumps(DIMENSION_LABELS, ensure_ascii=False)
    safe_title = html.escape(title)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{safe_title}</title>
<style>
:root{{--ink:#17213b;--muted:#65708a;--paper:#f5f7fc;--card:#fff;--accent:#6c5ce7;--accent2:#00b894}}
*{{box-sizing:border-box}} body{{margin:0;background:radial-gradient(circle at 90% 0,#e5ddff 0,transparent 30%),var(--paper);color:var(--ink);font:15px/1.45 system-ui,sans-serif}}
main{{width:min(1120px,calc(100% - 32px));margin:48px auto}} header{{margin-bottom:28px}} h1{{font-size:clamp(30px,5vw,52px);letter-spacing:-.04em;margin:0}} .eyebrow{{color:var(--accent);font-weight:750;text-transform:uppercase;letter-spacing:.12em}} .subtitle,.note{{color:var(--muted)}}
.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:28px 0}} .metric,.panel{{background:color-mix(in srgb,var(--card) 94%,transparent);border:1px solid #e3e7f1;border-radius:18px;box-shadow:0 12px 35px #2632580d}} .metric{{padding:20px}} .metric strong{{display:block;font-size:clamp(22px,3vw,34px);letter-spacing:-.04em}} .metric span{{color:var(--muted)}}
.panel{{padding:24px}} .toolbar{{display:flex;align-items:center;justify-content:space-between;gap:16px;margin-bottom:22px}} h2{{margin:0;font-size:23px}} select{{max-width:270px;padding:10px 34px 10px 12px;border:1px solid #d8ddeb;border-radius:10px;background:white;color:var(--ink);font:inherit}}
.chart{{display:grid;gap:13px}} .bar-row{{display:grid;grid-template-columns:minmax(105px,220px) 1fr 70px;gap:12px;align-items:center}} .label{{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}} .track{{height:14px;background:#eceffa;border-radius:99px;overflow:hidden}} .bar{{height:100%;min-width:2px;border-radius:99px;background:linear-gradient(90deg,var(--accent),#9b7cff 65%,var(--accent2));transform-origin:left;animation:grow .55s ease-out}} .value{{font-variant-numeric:tabular-nums;text-align:right;font-weight:700}} .empty{{padding:60px 20px;text-align:center;color:var(--muted)}}
@keyframes grow{{from{{transform:scaleX(0)}}}} @media(max-width:700px){{main{{margin-top:28px}}.metrics{{grid-template-columns:repeat(2,1fr)}}.toolbar{{align-items:flex-start;flex-direction:column}}.bar-row{{grid-template-columns:100px 1fr 52px}}}} @media(prefers-reduced-motion:reduce){{.bar{{animation:none}}}}
</style></head><body><main>
<header><div class="eyebrow">Hugging Face open weights</div><h1>{safe_title}</h1><p class="subtitle">An interactive overview generated from the catalogue's aggregate statistics.</p></header>
<section class="metrics" id="metrics" aria-label="Headline totals"></section>
<section class="panel"><div class="toolbar"><div><h2 id="chart-title">Distribution</h2><div class="note" id="chart-note"></div></div><label>Explore <select id="dimension" aria-label="Choose a dimension"></select></label></div><div class="chart" id="chart"></div></section>
</main><script>
const stats={payload}; const names={labels}; const topN={top};
const format=new Intl.NumberFormat();
const metricData=[["Models",stats.model_count],["Downloads",stats.downloads_total],["Likes",stats.likes_total],["Followers",stats.followers_total]];
document.querySelector("#metrics").innerHTML=metricData.map(([k,v])=>`<article class="metric"><strong>${{format.format(v)}}</strong><span>${{k}}</span></article>`).join("");
const select=document.querySelector("#dimension"); Object.keys(stats.dimensions).forEach(key=>{{const option=document.createElement("option");option.value=key;option.textContent=names[key]||key.replaceAll("_"," ");select.append(option)}});
function draw(){{const key=select.value, values=Object.entries(stats.dimensions[key]||{{}}).sort((a,b)=>b[1]-a[1]||a[0].localeCompare(b[0])).slice(0,topN), max=Math.max(1,...values.map(x=>x[1])); document.querySelector("#chart-title").textContent=names[key]||key; document.querySelector("#chart-note").textContent=`Top ${{values.length}} of ${{Object.keys(stats.dimensions[key]||{{}}).length}} groups`;
 const chart=document.querySelector("#chart");chart.replaceChildren(); if(!values.length){{chart.innerHTML='<div class="empty">No data for this dimension.</div>';return}} values.forEach(([label,value])=>{{const row=document.createElement("div");row.className="bar-row";const name=document.createElement("div");name.className="label";name.title=label;name.textContent=label;const track=document.createElement("div");track.className="track";const bar=document.createElement("div");bar.className="bar";bar.style.width=`${{100*value/max}}%`;track.append(bar);const number=document.createElement("div");number.className="value";number.textContent=format.format(value);row.append(name,track,number);chart.append(row)}})}}
select.addEventListener("change",draw);draw();
</script></body></html>"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate an HTML visualization from catalogue statistics JSON.")
    parser.add_argument("input", nargs="?", default="hf-open-weights-stats.json", help="statistics JSON input")
    parser.add_argument("-o", "--output", default="hf-open-weights-stats.html", help="HTML output path")
    parser.add_argument("--title", default="Model landscape", help="dashboard title")
    parser.add_argument("--top", type=int, default=15, help="maximum groups displayed per dimension")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        data = load_statistics(Path(args.input))
        output = Path(args.output)
        output.write_text(render_dashboard(data, args.title, args.top), encoding="utf-8")
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
    print(f"Visualization written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
