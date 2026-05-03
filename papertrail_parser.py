#!/usr/bin/env python3
"""
Papertrail log parser — supports .json, .json.gz, .tsv.gz
Handles files of any size (including 10+ GB) via streaming.

Usage:
    python papertrail_parser.py <file> [options]

Options:
    --format {table,csv,json}   Output format (default: table)
    --filter PATTERN            Filter messages by regex pattern
    --severity LEVEL            Filter by severity (e.g. error, warning, info)
    --hostname HOST             Filter by hostname
    --program PROGRAM           Filter by program name
    --since DATETIME            Show logs after this time (e.g. "2026-04-09 10:00:00")
    --until DATETIME            Show logs before this time
    --limit N                   Limit number of results (default: 500)
    --output FILE               Write output to file instead of stdout
"""

import gzip
import json
import csv
import re
import sys
import argparse
from datetime import datetime

try:
    import ijson
    HAS_IJSON = True
except ImportError:
    HAS_IJSON = False


# ── Parsers ──────────────────────────────────────────────────────────────────

def parse_tsv_gz(filepath):
    """Parse Papertrail TSV.gz archive (standard format)."""
    fields = ["id", "received_at", "display_received_at", "source_ip",
              "facility", "severity", "program", "message", "hostname"]
    events = []
    with gzip.open(filepath, "rt", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, fieldnames=fields, delimiter="\t")
        for row in reader:
            events.append(dict(row))
    return events


def _open_file(filepath):
    """Return an open text stream for .gz or plain files."""
    if filepath.endswith(".gz"):
        return gzip.open(filepath, "rt", encoding="utf-8", errors="replace")
    return open(filepath, "r", encoding="utf-8", errors="replace")


def _stream_json_array(filepath):
    """Stream a JSON array from a large file using ijson — O(1) memory."""
    if not HAS_IJSON:
        print("WARNING: ijson not installed. Loading full JSON array into RAM.\n"
              "  For large files run: pip install ijson", file=sys.stderr)
        with _open_file(filepath) as f:
            for item in json.load(f):
                yield item
        return

    open_fn = gzip.open if filepath.endswith(".gz") else open
    with open_fn(filepath, "rb") as f:
        for item in ijson.items(f, "item"):
            yield item


def _stream_ndjson(filepath):
    """Stream a newline-delimited JSON file line by line — O(1) memory."""
    with _open_file(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    pass


def parse_json(filepath, limit=0):
    """
    Parse JSON or JSON.gz — streams for any file size.
    Supports newline-delimited JSON (NDJSON) and JSON arrays.
    Stops early if limit > 0.
    """
    events = []
    count = 0

    # Peek at first byte to detect format
    with _open_file(filepath) as f:
        first_char = f.read(1)

    source = _stream_json_array(filepath) if first_char == "[" else _stream_ndjson(filepath)
    fmt    = "JSON array" if first_char == "[" else "NDJSON"
    print(f"Detected {fmt} — streaming ...", file=sys.stderr)

    for raw in source:
        events.append(normalize_event(raw))
        count += 1
        if count % 100_000 == 0:
            print(f"  ... {count:,} events read", file=sys.stderr)
        if limit and count >= limit:
            print(f"  Reached limit of {limit:,} — stopping early.", file=sys.stderr)
            break

    return events


# keep old name as alias
parse_json_gz = parse_json


def normalize_event(raw: dict) -> dict:
    """Normalize SolarWinds Papertrail JSON fields to a common schema."""
    syslog = raw.get("syslog") or {}
    return {
        "id":           raw.get("event_id", ""),
        "received_at":  syslog.get("timestamp") or raw.get("receive_time", ""),
        "source_ip":    raw.get("sender_ip_str") or raw.get("sw.remote.ip", ""),
        "facility":     syslog.get("facility", ""),
        "severity":     syslog.get("severity", ""),
        "program":      syslog.get("appName") or raw.get("syslog_appname", ""),
        "message":      raw.get("logmsg") or raw.get("syslog_message", ""),
        "hostname":     syslog.get("host") or raw.get("source_name", ""),
    }


def load_file(filepath, limit=0):
    """Auto-detect format (.json, .json.gz, .tsv.gz) and load events."""
    if filepath.endswith(".tsv.gz"):
        return parse_tsv_gz(filepath)
    return parse_json(filepath, limit=limit)


# ── Filtering ─────────────────────────────────────────────────────────────────

def parse_dt(s):
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unrecognised datetime format: {s}")


def apply_filters(events, args):
    filtered = []
    pattern = re.compile(args.filter, re.IGNORECASE) if args.filter else None
    since = parse_dt(args.since) if args.since else None
    until = parse_dt(args.until) if args.until else None

    for e in events:
        if args.hostname and args.hostname.lower() not in e.get("hostname", "").lower():
            continue
        if args.program and args.program.lower() not in e.get("program", "").lower():
            continue
        if args.severity and args.severity.lower() not in e.get("severity", "").lower():
            continue
        if pattern and not pattern.search(e.get("message", "")):
            continue
        if since or until:
            ts_raw = e.get("received_at", "")
            try:
                ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00")).replace(tzinfo=None)
                if since and ts < since:
                    continue
                if until and ts > until:
                    continue
            except (ValueError, AttributeError):
                pass  # keep events with unparseable timestamps
        filtered.append(e)

    if args.limit:
        filtered = filtered[:args.limit]
    return filtered


# ── Output formatters ─────────────────────────────────────────────────────────

SEVERITY_COLORS = {
    "emergency": "\033[91m", "alert": "\033[91m", "critical": "\033[91m",
    "error":     "\033[91m",
    "warning":   "\033[93m", "warn": "\033[93m",
    "notice":    "\033[96m",
    "info":      "\033[92m",
    "debug":     "\033[37m",
}
RESET = "\033[0m"


def severity_color(sev):
    return SEVERITY_COLORS.get(sev.lower(), "") if sev else ""


def output_table(events, out):
    if not events:
        print("No matching log events.", file=out)
        return

    # Column widths
    ts_w   = max(len(e.get("display_received_at") or e.get("received_at", "")) for e in events)
    ts_w   = max(ts_w, 19)
    host_w = max((len(e.get("hostname", "")) for e in events), default=8)
    host_w = max(host_w, 8)
    prog_w = max((len(e.get("program", "")) for e in events), default=7)
    prog_w = max(prog_w, 7)
    sev_w  = max((len(e.get("severity", "")) for e in events), default=8)
    sev_w  = max(sev_w, 8)

    sep = f"+{'-'*(ts_w+2)}+{'-'*(host_w+2)}+{'-'*(prog_w+2)}+{'-'*(sev_w+2)}+{'-'*62}+"
    hdr = (f"| {'Timestamp':<{ts_w}} | {'Hostname':<{host_w}} | "
           f"{'Program':<{prog_w}} | {'Severity':<{sev_w}} | {'Message':<60} |")

    print(sep, file=out)
    print(hdr, file=out)
    print(sep, file=out)

    for e in events:
        ts      = (e.get("display_received_at") or e.get("received_at", ""))[:ts_w]
        host    = e.get("hostname", "")[:host_w]
        prog    = e.get("program",  "")[:prog_w]
        sev     = e.get("severity", "")[:sev_w]
        msg     = e.get("message",  "").replace("\n", " ")

        color   = severity_color(sev)
        msg_col = 60
        # Print long messages across multiple rows
        while msg:
            chunk = msg[:msg_col]
            msg   = msg[msg_col:]
            line  = (f"| {ts:<{ts_w}} | {host:<{host_w}} | {prog:<{prog_w}} | "
                     f"{color}{sev:<{sev_w}}{RESET} | {chunk:<{msg_col}} |")
            print(line, file=out)
            ts = host = prog = sev = ""   # blank continuation rows

    print(sep, file=out)
    print(f"\n{len(events)} event(s) shown.", file=out)


def output_csv(events, out):
    if not events:
        print("No matching log events.", file=out)
        return
    fieldnames = ["received_at", "hostname", "program", "severity", "source_ip",
                  "facility", "message", "id"]
    writer = csv.DictWriter(out, fieldnames=fieldnames, extrasaction="ignore",
                            lineterminator="\n")
    writer.writeheader()
    writer.writerows(events)


def output_json(events, out):
    json.dump(events, out, indent=2, ensure_ascii=False)
    print(file=out)


def output_html(events, source_file, out):
    """Generate a self-contained HTML page with live search, filters, and color-coded severity."""
    SEV_BADGE = {
        "emergency": "#dc2626", "alert": "#dc2626", "critical": "#dc2626",
        "error":     "#ef4444",
        "warning":   "#f59e0b", "warn": "#f59e0b",
        "notice":    "#06b6d4",
        "info":      "#22c55e",
        "debug":     "#94a3b8",
    }

    def badge_color(sev):
        return SEV_BADGE.get((sev or "").lower(), "#6b7280")

    def esc(s):
        s = str(s) if s is not None else ""
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

    rows_html = []
    severities = sorted({e.get("severity", "") for e in events if e.get("severity")})
    hostnames  = sorted({e.get("hostname",  "") for e in events if e.get("hostname")})
    programs   = sorted({e.get("program",   "") for e in events if e.get("program")})

    for e in events:
        ts   = esc(e.get("display_received_at") or e.get("received_at", ""))
        host = esc(e.get("hostname", ""))
        prog = esc(e.get("program",  ""))
        sev  = esc(e.get("severity", ""))
        ip   = esc(e.get("source_ip",""))
        msg  = esc(e.get("message",  ""))
        color = badge_color(sev)
        rows_html.append(
            f'<tr data-sev="{sev.lower()}" data-host="{host.lower()}" data-prog="{prog.lower()}">'
            f'<td class="ts">{ts}</td>'
            f'<td><span class="host">{host}</span></td>'
            f'<td><span class="prog">{prog}</span></td>'
            f'<td><span class="badge" style="background:{color}">{sev or "—"}</span></td>'
            f'<td class="msg">{msg}</td>'
            f'<td class="ip">{ip}</td>'
            f'</tr>'
        )

    sev_options  = "".join(f'<option value="{s}">{s}</option>' for s in severities)
    host_options = "".join(f'<option value="{h}">{h}</option>' for h in hostnames)
    prog_options = "".join(f'<option value="{p}">{p}</option>' for p in programs)
    rows_joined  = "\n".join(rows_html)
    total        = len(events)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Papertrail Logs — {esc(source_file)}</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0f172a; color: #e2e8f0; font-size: 13px; }}
  header {{ background: #1e293b; padding: 16px 24px; border-bottom: 1px solid #334155; display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }}
  header h1 {{ font-size: 16px; font-weight: 600; color: #f8fafc; flex: 1 1 auto; }}
  .controls {{ display: flex; gap: 8px; flex-wrap: wrap; align-items: center; width: 100%; margin-top: 10px; }}
  input, select {{ background: #0f172a; border: 1px solid #334155; color: #e2e8f0; padding: 6px 10px; border-radius: 6px; font-size: 12px; outline: none; }}
  input:focus, select:focus {{ border-color: #6366f1; }}
  input#search {{ flex: 1 1 240px; }}
  #count {{ font-size: 12px; color: #94a3b8; white-space: nowrap; }}
  .table-wrap {{ overflow-x: auto; padding: 0 8px 40px; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
  th {{ background: #1e293b; color: #94a3b8; text-transform: uppercase; font-size: 11px; letter-spacing: .05em; padding: 8px 10px; text-align: left; position: sticky; top: 0; z-index: 1; border-bottom: 1px solid #334155; }}
  td {{ padding: 7px 10px; border-bottom: 1px solid #1e293b; vertical-align: top; }}
  tr:hover td {{ background: #1e293b; }}
  .ts  {{ color: #94a3b8; white-space: nowrap; font-size: 11px; }}
  .ip  {{ color: #64748b; font-size: 11px; }}
  .msg {{ word-break: break-word; max-width: 680px; line-height: 1.5; color: #cbd5e1; }}
  .host {{ background: #1e3a5f; color: #93c5fd; padding: 2px 7px; border-radius: 4px; font-size: 11px; }}
  .prog {{ background: #1e3a2e; color: #86efac; padding: 2px 7px; border-radius: 4px; font-size: 11px; }}
  .badge {{ color: #fff; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; display: inline-block; }}
  .hidden {{ display: none; }}
  .highlight {{ background: #854d0e; border-radius: 2px; padding: 0 2px; }}
</style>
</head>
<body>
<header>
  <h1>Papertrail Logs &nbsp;·&nbsp; <span style="color:#94a3b8;font-weight:400">{esc(source_file)}</span></h1>
  <div class="controls">
    <input id="search" type="search" placeholder="Search messages..." oninput="applyFilters()">
    <select id="fSev"  onchange="applyFilters()"><option value="">All severities</option>{sev_options}</select>
    <select id="fHost" onchange="applyFilters()"><option value="">All hosts</option>{host_options}</select>
    <select id="fProg" onchange="applyFilters()"><option value="">All programs</option>{prog_options}</select>
    <span id="count">{total} events</span>
  </div>
</header>
<div class="table-wrap">
<table id="logtable">
  <thead><tr>
    <th>Timestamp</th><th>Hostname</th><th>Program</th>
    <th>Severity</th><th>Message</th><th>Source IP</th>
  </tr></thead>
  <tbody id="tbody">
{rows_joined}
  </tbody>
</table>
</div>
<script>
function applyFilters() {{
  const q    = document.getElementById('search').value.toLowerCase();
  const sev  = document.getElementById('fSev').value.toLowerCase();
  const host = document.getElementById('fHost').value.toLowerCase();
  const prog = document.getElementById('fProg').value.toLowerCase();
  const rows = document.querySelectorAll('#tbody tr');
  let visible = 0;
  rows.forEach(r => {{
    const msgCell = r.querySelector('.msg');
    const msgText = msgCell.textContent.toLowerCase();
    const show = (!q    || msgText.includes(q))
              && (!sev  || r.dataset.sev  === sev)
              && (!host || r.dataset.host === host)
              && (!prog || r.dataset.prog === prog);
    r.classList.toggle('hidden', !show);
    if (show) {{
      visible++;
      // highlight search term
      if (q) {{
        const raw = msgCell.textContent;
        const re  = new RegExp(q.replace(/[.*+?^${{}}()|[\\]\\\\]/g,'\\\\$&'), 'gi');
        msgCell.innerHTML = raw.replace(re, m => `<mark class="highlight">${{m}}</mark>`);
      }} else {{
        msgCell.textContent = msgCell.textContent; // reset
      }}
    }}
  }});
  document.getElementById('count').textContent = visible + ' / {total} events';
}}
</script>
</body>
</html>"""
    out.write(html)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Parse Papertrail log archives into a human-readable format."
    )
    parser.add_argument("file", help="Path to the log file (.json, .json.gz, .tsv.gz)")
    parser.add_argument("--format", choices=["html", "table", "csv", "json"], default="html",
                        help="Output format (default: html)")
    parser.add_argument("--filter",   metavar="PATTERN", help="Regex filter on message text")
    parser.add_argument("--severity", metavar="LEVEL",   help="Filter by severity (e.g. error)")
    parser.add_argument("--hostname", metavar="HOST",    help="Filter by hostname (partial match)")
    parser.add_argument("--program",  metavar="PROGRAM", help="Filter by program/app name")
    parser.add_argument("--since",    metavar="DATETIME",help="Show events after  (YYYY-MM-DD HH:MM:SS)")
    parser.add_argument("--until",    metavar="DATETIME",help="Show events before (YYYY-MM-DD HH:MM:SS)")
    parser.add_argument("--limit",    metavar="N", type=int, default=0,
                        help="Max number of results (0 = no limit, default: 0)")
    parser.add_argument("--output",   metavar="FILE", default="logs.html",
                        help="Output file (default: logs.html)")
    args = parser.parse_args()

    print(f"Loading {args.file} ...", file=sys.stderr)
    try:
        events = load_file(args.file, limit=args.limit)
    except Exception as exc:
        print(f"ERROR: could not read file — {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(events)} events. Applying filters ...", file=sys.stderr)
    events = apply_filters(events, args)
    print(f"{len(events)} events after filtering.", file=sys.stderr)

    # HTML goes to file by default; other formats go to stdout unless --output given
    if args.format == "html":
        out_path = args.output if args.output else "logs.html"
        with open(out_path, "w", encoding="utf-8") as f:
            output_html(events, args.file, f)
        print(f"HTML report written to {out_path} — open in any browser.", file=sys.stderr)
    else:
        out = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
        try:
            if args.format == "table":
                output_table(events, out)
            elif args.format == "csv":
                output_csv(events, out)
            else:
                output_json(events, out)
        finally:
            if args.output:
                out.close()
                print(f"Output written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
