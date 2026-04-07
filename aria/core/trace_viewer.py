"""Generate a standalone HTML trace viewer from a TaskTrace.

Produces a single self-contained HTML file with:
- Demo grids (input / output)
- Seeds panel (graph structure + specialization + provenance)
- Static pipeline attempts
- Deterministic editor results
- Learned editor results
- Graph diagrams (Mermaid)
- Timeline of events
"""

from __future__ import annotations

import json
from typing import Any

from aria.core.trace import TaskTrace


# ARC color palette (0-9)
_COLORS = [
    "#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00",
    "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25",
]


def _grid_html(grid: list[list[int]], cell_size: int = 22) -> str:
    """Render a grid as an HTML table with ARC colors."""
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    parts = [f'<table style="border-collapse:collapse;margin:4px 0;">']
    for r in range(rows):
        parts.append("<tr>")
        for c in range(cols):
            val = grid[r][c] if c < len(grid[r]) else 0
            color = _COLORS[val % len(_COLORS)]
            text_color = "#fff" if val in (0, 9) else "#000"
            parts.append(
                f'<td style="width:{cell_size}px;height:{cell_size}px;'
                f'background:{color};color:{text_color};text-align:center;'
                f'font-size:10px;border:1px solid #333;">{val}</td>'
            )
        parts.append("</tr>")
    parts.append("</table>")
    return "".join(parts)


def _diff_grid_html(input_grid: list[list[int]], output_grid: list[list[int]],
                    cell_size: int = 22) -> str:
    """Render a diff overlay: green=same, red=different."""
    rows = max(len(input_grid), len(output_grid))
    cols = max(
        (len(input_grid[0]) if input_grid else 0),
        (len(output_grid[0]) if output_grid else 0),
    )
    parts = [f'<table style="border-collapse:collapse;margin:4px 0;">']
    for r in range(rows):
        parts.append("<tr>")
        for c in range(cols):
            iv = input_grid[r][c] if r < len(input_grid) and c < len(input_grid[r]) else -1
            ov = output_grid[r][c] if r < len(output_grid) and c < len(output_grid[r]) else -1
            same = iv == ov
            bg = "#2a2" if same else "#c33"
            parts.append(
                f'<td style="width:{cell_size}px;height:{cell_size}px;'
                f'background:{bg};color:#fff;text-align:center;'
                f'font-size:10px;border:1px solid #333;">'
                f'{ov if ov >= 0 else "?"}</td>'
            )
        parts.append("</tr>")
    parts.append("</table>")
    return "".join(parts)


def _graph_mermaid(graph_dict: dict) -> str:
    """Generate a Mermaid graph definition from a graph dict."""
    lines = ["graph TD"]
    nodes = graph_dict.get("nodes", {})
    output_id = graph_dict.get("output_id", "")

    for nid, node in nodes.items():
        label = f"{nid}\\n{node['op']}"
        slots = node.get("slots", [])
        if slots:
            slot_str = ", ".join(f"{s['name']}={s.get('evidence','?')}" for s in slots[:3])
            label += f"\\n[{slot_str}]"
        style = ""
        if nid == output_id:
            style = ":::output"
        lines.append(f'    {nid}["{label}"]{style}')

    for nid, node in nodes.items():
        for inp in node.get("inputs", []):
            if inp == "input":
                lines.append(f'    INPUT(("input")) --> {nid}')
            elif inp in nodes:
                lines.append(f"    {inp} --> {nid}")

    lines.append("    classDef output fill:#4a4,stroke:#2a2,color:#fff")
    return "\n".join(lines)


def _bindings_table(spec_dict: dict | None) -> str:
    if not spec_dict or not spec_dict.get("bindings"):
        return "<em>no bindings</em>"
    parts = ['<table class="bindings"><tr><th>Node</th><th>Name</th><th>Value</th><th>Source</th></tr>']
    for b in spec_dict["bindings"]:
        val = str(b["value"])
        if len(val) > 60:
            val = val[:60] + "..."
        parts.append(
            f'<tr><td>{b["node_id"]}</td><td>{b["name"]}</td>'
            f'<td>{val}</td><td>{b.get("source","")}</td></tr>'
        )
    parts.append("</table>")
    return "".join(parts)


def _event_html(event: dict) -> str:
    phase = event.get("phase", "")
    etype = event.get("event_type", "")
    data = event.get("data", {})

    phase_colors = {
        "static": "#369",
        "seeds": "#963",
        "deterministic": "#639",
        "learned": "#693",
    }
    color = phase_colors.get(phase, "#666")

    parts = [f'<div class="event" style="border-left:3px solid {color};">']
    parts.append(f'<span class="phase-badge" style="background:{color};">{phase}</span>')
    parts.append(f'<strong>{etype}</strong>')

    # Render key data inline
    skip_keys = {"graph", "specialization", "compile_result"}
    inline = {k: v for k, v in data.items() if k not in skip_keys}
    if inline:
        parts.append(f'<span class="data">{json.dumps(inline, default=str)}</span>')

    # Graph snapshot
    if "graph" in data and data["graph"]:
        mermaid = _graph_mermaid(data["graph"])
        parts.append(f'<details><summary>Graph</summary>'
                     f'<pre class="mermaid">{mermaid}</pre></details>')

    # Specialization
    if "specialization" in data and data["specialization"]:
        parts.append(f'<details><summary>Specialization</summary>'
                     f'{_bindings_table(data["specialization"])}</details>')

    # Compile result
    if "compile_result" in data and data["compile_result"]:
        cr = data["compile_result"]
        status = cr.get("status", "unknown")
        badge = "ok" if status == "success" else "fail"
        parts.append(f'<span class="compile-{badge}">{status}</span>')
        if status == "failure":
            parts.append(f'<span class="reason">{cr.get("reason","")}</span>')

    parts.append("</div>")
    return "".join(parts)


def generate_html(trace: TaskTrace) -> str:
    """Generate a standalone HTML trace viewer."""
    d = trace.to_dict()

    # Demo grids
    demo_html = ""
    for i, demo in enumerate(d["demos"]):
        demo_html += f'<div class="demo"><h4>Demo {i}</h4>'
        demo_html += '<div class="grid-row">'
        demo_html += f'<div><div class="label">Input</div>{_grid_html(demo["input"])}</div>'
        demo_html += f'<div><div class="label">Output</div>{_grid_html(demo["output"])}</div>'
        demo_html += f'<div><div class="label">Diff</div>{_diff_grid_html(demo["input"], demo["output"])}</div>'
        demo_html += '</div></div>'

    # Seeds
    seeds_html = ""
    for i, seed in enumerate(d["seeds"]):
        seeds_html += f'<div class="seed">'
        seeds_html += f'<h4>Seed {i} ({seed["provenance"]})'
        if seed["verified"]:
            seeds_html += ' <span class="ok">VERIFIED</span>'
        seeds_html += '</h4>'
        mermaid = _graph_mermaid(seed["graph"])
        seeds_html += f'<pre class="mermaid">{mermaid}</pre>'
        seeds_html += _bindings_table(seed.get("specialization"))
        seeds_html += '</div>'

    # Events timeline
    events_html = ""
    for event in d["events"]:
        events_html += _event_html(event)

    # Results summary
    summary_parts = []
    for phase, result in [("Static", d.get("static_result", {})),
                          ("Deterministic", d.get("deterministic_result", {})),
                          ("Learned", d.get("learned_result", {}))]:
        if not result:
            continue
        solved = result.get("solved", False)
        badge = '<span class="ok">SOLVED</span>' if solved else '<span class="fail">UNSOLVED</span>'
        summary_parts.append(f'<div class="result-row"><strong>{phase}</strong> {badge} '
                             f'{json.dumps({k:v for k,v in result.items() if k != "solved"}, default=str)}</div>')
    results_html = "".join(summary_parts)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Trace: {d["task_id"]}</title>
<style>
body {{ font-family: monospace; background: #1a1a1a; color: #ddd; margin: 20px; }}
h1, h2, h3, h4 {{ color: #fff; margin: 8px 0; }}
.section {{ background: #252525; border-radius: 6px; padding: 12px; margin: 12px 0; }}
.grid-row {{ display: flex; gap: 16px; flex-wrap: wrap; align-items: start; }}
.label {{ font-size: 11px; color: #888; margin-bottom: 2px; }}
.demo {{ margin: 8px 0; }}
.seed {{ background: #2a2a2a; padding: 8px; margin: 6px 0; border-radius: 4px; }}
.event {{ padding: 6px 10px; margin: 3px 0; background: #222; border-radius: 3px; font-size: 12px; }}
.phase-badge {{ display: inline-block; padding: 1px 6px; border-radius: 3px; color: #fff; font-size: 10px; margin-right: 6px; }}
.data {{ color: #888; font-size: 11px; margin-left: 8px; }}
.ok {{ color: #2ECC40; font-weight: bold; }}
.fail {{ color: #FF4136; font-weight: bold; }}
.compile-ok {{ color: #2ECC40; margin-left: 8px; }}
.compile-fail {{ color: #FF4136; margin-left: 8px; }}
.reason {{ color: #FF851B; font-size: 11px; margin-left: 4px; }}
.result-row {{ padding: 6px; margin: 4px 0; }}
table.bindings {{ font-size: 11px; border-collapse: collapse; margin: 4px 0; }}
table.bindings th, table.bindings td {{ border: 1px solid #444; padding: 2px 6px; }}
table.bindings th {{ background: #333; }}
pre.mermaid {{ background: #1e1e1e; padding: 8px; border-radius: 4px; font-size: 11px; overflow-x: auto; }}
details {{ margin: 4px 0; }}
summary {{ cursor: pointer; color: #7FDBFF; font-size: 11px; }}
.banner {{ padding: 8px 12px; border-radius: 4px; font-size: 14px; margin: 8px 0; }}
.banner.solved {{ background: #1a3a1a; border: 1px solid #2ECC40; }}
.banner.unsolved {{ background: #3a1a1a; border: 1px solid #FF4136; }}
</style>
</head>
<body>

<h1>Task: {d["task_id"]}</h1>
<div class="banner {'solved' if d['solved'] else 'unsolved'}">
  {'SOLVED by ' + d.get('solver', '?') if d['solved'] else 'UNSOLVED'}
</div>

<div class="section">
<h2>Demos ({len(d['demos'])})</h2>
{demo_html}
</div>

<div class="section">
<h2>Results</h2>
{results_html}
</div>

<div class="section">
<h2>Seeds ({len(d['seeds'])})</h2>
{seeds_html}
</div>

<div class="section">
<h2>Timeline ({len(d['events'])} events)</h2>
{events_html}
</div>

<script>
// Mermaid rendering (optional — works if included, plain text if not)
</script>
</body>
</html>"""
