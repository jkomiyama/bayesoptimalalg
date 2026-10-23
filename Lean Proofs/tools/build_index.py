#!/usr/bin/env python3
"""Build offline navigation for the Lean sources, using Python's standard library.

This is an administrative source index, not a mathematical verifier. It does not
run Lean, establish manuscript coverage, or change any Lean source file. The
scanned public theorem/lemma names must match declarations.json, the inventory
used by the separately run Lean aggregate audit.

Run from any directory: python path/to/junpei_lean_submission/tools/build_index.py
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import html
import json
from pathlib import Path
import re
import sys
from urllib.parse import quote


@dataclass(frozen=True)
class Declaration:
    name: str
    kind: str
    file: str
    line: int
    module: str
    module_summary: str


@dataclass(frozen=True)
class Module:
    name: str
    file: str
    summary: str
    imports: tuple[str, ...]
    declarations: tuple[Declaration, ...]


def mask_comments_and_strings(source: str) -> tuple[str, list[tuple[int, str]]]:
    """Preserve offsets and line breaks while masking nested comments/strings.

    Return module documentation comments separately. This scanner is intended
    for the source syntax in this archive; Lean remains the source parser and
    proof checker used by verify.py.
    """
    result = list(source)
    module_docs: list[tuple[int, str]] = []
    index = 0
    while index < len(source):
        start = index
        if source.startswith("--", index):
            newline = source.find("\n", index)
            index = len(source) if newline == -1 else newline
        elif source.startswith("/-", index):
            is_module_doc = source.startswith("/-!", index)
            depth = 1
            index += 2
            while index < len(source) and depth:
                if source.startswith("/-", index):
                    depth += 1
                    index += 2
                elif source.startswith("-/", index):
                    depth -= 1
                    index += 2
                else:
                    index += 1
            if depth:
                raise ValueError("Unterminated block comment")
            if is_module_doc:
                module_docs.append((start, source[start + 3:index - 2]))
        elif source[index] == '"':
            index += 1
            while index < len(source):
                if source[index] == "\\":
                    index += 2
                elif source[index] == '"':
                    index += 1
                    break
                else:
                    index += 1
        else:
            index += 1
            continue
        for position in range(start, min(index, len(source))):
            if result[position] not in "\r\n":
                result[position] = " "
    return "".join(result), module_docs


COMMAND = re.compile(
    r"^[ \t]*(?:@\[[^\]]*\][ \t\r\n]*)*"
    r"(?P<mods>(?:(?:private|protected|noncomputable|unsafe)\s+)*)"
    r"(?P<kind>namespace|section|end|theorem|lemma)\b"
    r"[ \t]*(?P<name>[\w'.]+)?",
    re.MULTILINE,
)


def scan_module(path: Path, root: Path) -> Module:
    source = path.read_text(encoding="utf-8-sig")
    cleaned, module_docs = mask_comments_and_strings(source)
    relative = path.relative_to(root).as_posix()
    module_name = relative.removesuffix(".lean").replace("/", ".")
    summary = " ".join(module_docs[0][1].split()) if module_docs else ""
    imports = tuple(re.findall(r"^import\s+([\w.]+)", cleaned, re.MULTILINE))
    # Each scope records its kind, declared name, and complete namespace.
    scopes: list[tuple[str, str, str]] = []
    declarations: list[Declaration] = []
    for match in COMMAND.finditer(cleaned):
        kind = match.group("kind")
        name = match.group("name") or ""
        current_namespace = scopes[-1][2] if scopes else ""
        if kind == "namespace":
            if not name:
                raise ValueError(f"Missing namespace name in {relative}")
            prefix = (name[len("_root_."):] if name.startswith("_root_.")
                      else ".".join(filter(None, (current_namespace, name))))
            scopes.append((kind, name, prefix))
        elif kind == "section":
            scopes.append((kind, name, current_namespace))
        elif kind == "end":
            if not scopes:
                raise ValueError(f"Unexpected end in {relative}")
            scope_kind, scope_name, _ = scopes.pop()
            if name and name != scope_name:
                raise ValueError(
                    f"Mismatched end {name!r} for {scope_kind} {scope_name!r} in {relative}"
                )
        elif "private" not in match.group("mods").split():
            if not name:
                raise ValueError(f"Missing {kind} name in {relative}")
            full_name = (name[len("_root_."):] if name.startswith("_root_.")
                         else ".".join(filter(None, (current_namespace, name))))
            # Use the keyword position, rather than a preceding attribute line.
            line = cleaned.count("\n", 0, match.start("kind")) + 1
            declarations.append(Declaration(
                full_name, kind, relative, line, module_name, summary
            ))
    if scopes:
        raise ValueError(f"Unclosed namespace/section in {relative}: {scopes}")
    return Module(module_name, relative, summary, imports, tuple(declarations))


def write_tsv(path: Path, declarations: list[Declaration]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(("declaration", "kind", "file", "line", "module", "module_header_summary"))
        for declaration in declarations:
            writer.writerow((declaration.name, declaration.kind, declaration.file,
                             declaration.line, declaration.module, declaration.module_summary))


def write_modules(path: Path, modules: list[Module]) -> None:
    lines = [
        "LEAN SOURCE MODULE INDEX",
        "========================",
        "",
        f"{len(modules)} modules; {sum(len(m.declarations) for m in modules)} public theorem/lemma declarations.",
        "Paths are relative to the project root. Imports are direct Lean imports.",
        "Descriptions reproduce module documentation headers where present.",
        "A missing header is reported explicitly; it is not inferred from a filename.",
        "This is navigation only. See PROOF_MAP.txt for manuscript coverage and",
        "root REPORT.txt, verification_results.json, and logs/ for recorded Lean",
        "evidence; run verify.py to recheck proofs.",
        "",
    ]
    for module in modules:
        lines.extend((
            module.name,
            "  Source: " + module.file,
            f"  Public theorem/lemma declarations: {len(module.declarations)}",
            "  Header: " + (module.summary or "[No module documentation header in this source.]"),
            "  Imports: " + ", ".join(module.imports),
            "",
        ))
    path.write_text("\n".join(lines), encoding="utf-8")


def write_html(path: Path, declarations: list[Declaration], modules: list[Module]) -> None:
    rows = []
    for declaration in declarations:
        link = "../" + quote(declaration.file, safe="/")
        summary = declaration.module_summary or "No module documentation header."
        rows.append(
            '<tr><td><code>' + html.escape(declaration.name) + '</code></td>'
            '<td><a href="' + link + '">' + html.escape(declaration.file) + '</a>'
            '<br><span class="line">line ' + str(declaration.line) + '</span></td>'
            '<td class="summary">' + html.escape(summary) + '</td></tr>'
        )
    document = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Junpei Lean declaration index</title>
<style>
:root{color-scheme:light dark;font-family:system-ui,sans-serif;line-height:1.5}
body{max-width:1500px;margin:0 auto;padding:1.5rem;color:#16212d;background:#fff}
h1{font-size:1.8rem;margin:.2rem 0}.intro{max-width:80ch}a{color:#164f9c}
.controls{position:sticky;top:0;background:#fff;padding:.75rem 0;border-bottom:1px solid #bec9d3}
label{display:block;font-weight:600}input{box-sizing:border-box;width:100%;max-width:750px;
font:inherit;padding:.6rem;border:1px solid #91a0af;border-radius:4px}
#count{margin:.4rem 0;font-variant-numeric:tabular-nums}table{width:100%;border-collapse:collapse;
text-align:left;font-size:.9rem}th,td{padding:.65rem;vertical-align:top;border-bottom:1px solid #d8dfe6}
th{background:#eef3f7}code{overflow-wrap:anywhere;font-size:.95em}.line{color:#4a5b6d}
.summary{max-width:48ch}td:nth-child(2){overflow-wrap:anywhere}tr[hidden]{display:none}
@media(max-width:800px){.summary,th:last-child{display:none}body{padding:.75rem}table{font-size:.8rem}}
@media(prefers-color-scheme:dark){body,.controls{background:#141b22;color:#e8eef5}a{color:#92c3ff}
th{background:#25313e}th,td,.controls{border-color:#3d4e60}.line{color:#bdcbd9}}
</style>
</head>
<body>
<h1>Junpei Lean declaration index</h1>
<div class="intro">
<p><strong>__TOTAL__ public theorem and lemma declarations in __MODULES__ proof modules.</strong>
This offline page helps locate declarations. Mathematical acceptance comes from Lean;
this generated index does not verify proofs or establish manuscript coverage.</p>
<p>Search a full or partial declaration name, a source filename, or words in a module header.
All search words must match the same row. Source links open the corresponding <code>.lean</code>
file; use the displayed line number in your Lean editor. A browser may download raw source files.</p>
<p><a href="PROOF_MAP.html">Manuscript proof map</a> ·
<a href="PROOF_MAP.txt">Plain-text proof map</a> ·
<a href="MODULES.txt">Module index</a> ·
<a href="declarations.tsv">TSV index</a> ·
<a href="../README.txt">Usage instructions</a></p>
</div>
<div class="controls">
<label for="search">Filter declarations</label>
<input id="search" type="search" placeholder="e.g. RealPrecisionMain theorem2" autocomplete="off">
<p id="count" role="status" aria-live="polite">__TOTAL__ of __TOTAL__ declarations shown</p>
</div>
<noscript><p>JavaScript is disabled. Every declaration is shown; use your browser's Find command.</p></noscript>
<table><thead><tr><th scope="col">Full Lean name</th><th scope="col">Source</th>
<th scope="col">Module documentation header</th></tr></thead><tbody>
__ROWS__
</tbody></table>
<script>
"use strict";
const rows = Array.from(document.querySelectorAll("tbody tr"));
const text = rows.map(row => row.textContent.toLocaleLowerCase());
const field = document.getElementById("search");
const count = document.getElementById("count");
field.addEventListener("input", () => {
  const words = field.value.toLocaleLowerCase().trim().split(/\s+/).filter(Boolean);
  let shown = 0;
  rows.forEach((row, index) => {
    const matches = words.every(word => text[index].includes(word));
    row.hidden = !matches;
    if (matches) shown++;
  });
  count.textContent = `${shown.toLocaleString()} of ${rows.length.toLocaleString()} declarations shown`;
});
</script>
</body></html>
'''
    document = document.replace("__TOTAL__", f"{len(declarations):,}")
    document = document.replace("__MODULES__", str(len(modules)))
    document = document.replace("__ROWS__", "\n".join(rows))
    path.write_text(document, encoding="utf-8")


def write_proof_map(root: Path) -> int:
    """Render the existing coverage TSV without inferring mathematical status."""
    with (root / "docs" / "proof_map.tsv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {"paper_item", "kind", "pdf_pages", "status", "title", "scope",
                    "lean_declarations", "lean_source_locations", "lean_declaration_kinds"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("docs/proof_map.tsv is missing required coverage columns")
        items = list(reader)
    if not items:
        raise ValueError("docs/proof_map.tsv contains no paper items")
    cards = []
    references = 0
    for index, item in enumerate(items, 1):
        status = item["status"]
        if "REFUTED" in status:
            category = "refuted"
        elif "UNFORMALIZED" in status or "OUTSIDE" in status:
            category = "outside"
        elif status in {"OPEN", "EXPLICITLY UNPROVED"}:
            category = "open"
        elif status.startswith("PROVED") or status.startswith("FORMAL PROOF"):
            category = "proved"
        else:
            category = "definition"
        page_links = re.sub(
            r"\d+",
            lambda match: '<a href="../manuscript/junpei_version.pdf#page='
                          + match.group() + '">' + match.group() + '</a>',
            html.escape(item["pdf_pages"]),
        )
        names = item["lean_declarations"].split()
        locations = [x.strip() for x in item["lean_source_locations"].split(";") if x.strip()]
        kinds = [x.strip() for x in item["lean_declaration_kinds"].split(";") if x.strip()]
        if not len(names) == len(locations) == len(kinds):
            raise ValueError(f"Unaligned declaration references for {item['paper_item']}")
        source_links = []
        for name, location, kind in zip(names, locations, kinds):
            source, separator, line = location.rpartition(":")
            path = root / source
            if (not separator or not line.isdecimal() or int(line) < 1
                    or not path.resolve().is_relative_to(root)
                    or not path.is_file() or path.suffix != ".lean"):
                raise ValueError(f"Invalid Lean source location: {location}")
            source_links.append(
                '<li><a href="../' + quote(source, safe="/") + '"><code>'
                + html.escape(name) + '</code></a><br><span class="location">'
                + html.escape(kind + " · " + location) + '</span></li>'
            )
        references += len(source_links)
        source_list = ('<ul class="sources">' + "".join(source_links) + '</ul>'
                       if source_links else '<p class="no-source">No Lean declaration is claimed for this item.</p>')
        cards.append(
            '<article id="paper-item-' + str(index) + '" class="item ' + category
            + '" data-status="' + html.escape(status, quote=True)
            + '" data-kind="' + html.escape(item["kind"], quote=True) + '">'
            '<div class="item-top"><h2>' + html.escape(item["paper_item"]) + '</h2>'
            '<span class="status">' + html.escape(status) + '</span></div>'
            '<p class="metadata">' + html.escape(item["kind"]) + ' · PDF pages ' + page_links + '</p>'
            '<h3>' + html.escape(item["title"]) + '</h3>'
            '<p>' + html.escape(item["scope"]) + '</p>' + source_list + '</article>'
        )
    def options(key: str) -> str:
        return "\n".join('<option value="' + html.escape(value, quote=True) + '">'
                         + html.escape(value) + '</option>' for value in sorted({item[key] for item in items}))
    document = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Junpei manuscript to Lean proof map</title>
<style>
:root{color-scheme:light dark;font-family:system-ui,sans-serif;line-height:1.55}
body{max-width:1150px;margin:0 auto;padding:1.5rem;color:#16212d;background:#fff}
h1{font-size:1.8rem;margin:.2rem 0}h2{font-size:1.25rem;margin:0}h3{font-size:1.05rem;margin:.6rem 0}
a{color:#164f9c}.intro{max-width:90ch}.notice{padding:.8rem 1rem;border-left:4px solid #9c2949;background:#fff2f5}
.controls{position:sticky;top:0;z-index:1;background:#fff;padding:.75rem 0;border-bottom:1px solid #bec9d3}
.fields{display:flex;gap:.7rem;flex-wrap:wrap}.field{display:flex;flex-direction:column;flex:1;min-width:13rem}
.field.search{flex:2}label{font-weight:600}input,select{box-sizing:border-box;width:100%;font:inherit;
padding:.55rem;border:1px solid #91a0af;border-radius:4px}select{font-size:.9rem}
#count{margin:.4rem 0;font-variant-numeric:tabular-nums}.item{margin:1.1rem 0;padding:1rem 1.2rem;
border:1px solid #cbd5df;border-left:5px solid #426a9a;border-radius:5px;scroll-margin-top:10rem}
.item-top{display:flex;justify-content:space-between;gap:1rem;flex-wrap:wrap}
.status{font-size:.8rem;font-weight:700;letter-spacing:.02em;border-radius:4px;
padding:.2rem .5rem;background:#edf3fa;color:#234d7f}.proved{border-left-color:#25704a}
.proved .status{color:#155f3b;background:#e7f6ed}.refuted{border-left-color:#9c2949}
.refuted .status{color:#8b1738;background:#ffedf2}.outside{border-left-color:#956016}
.outside .status{color:#81500b;background:#fff3db}.open{border-left-color:#656b74}
.open .status{color:#4b5058;background:#edf0f3}.metadata,.location,.no-source{color:#4a5b6d}
.metadata{margin:.45rem 0;font-size:.9rem}.sources{padding-left:1.2rem}.sources li{margin:.55rem 0}
.location{font-size:.85rem}code{font-size:.95em;overflow-wrap:anywhere}article[hidden]{display:none}
@media(max-width:650px){body{padding:.75rem}.item{padding:.8rem}.controls{position:static}.field{min-width:100%}}
@media(prefers-color-scheme:dark){body,.controls{background:#141b22;color:#e8eef5}a{color:#92c3ff}
.notice{background:#341c26}.item,.controls{border-color:#3d4e60}.metadata,.location,.no-source{color:#bdcbd9}}
</style>
</head>
<body>
<h1>Manuscript to Lean proof map</h1>
<div class="intro">
<p><strong>__TOTAL__ paper items and __REFERENCES__ declaration references.</strong>
This guide connects the supplied manuscript to explicit Lean statements and definitions.
It is documentation of the translation; Lean performs the mathematical proof checking.</p>
<p class="notice"><strong>Read each status and its scope.</strong> All 18 numbered proved results have
formal counterparts. Footnote 14 is <strong>refuted</strong>: Lean accepts a counterexample to its
claimed necessary gap scale. The separate Lemma 19 counterexample refutes a stronger tie-rule
interpretation, not Lemma 19. Open statements and unformalized context are labeled explicitly.
A successful build does not certify every assertion in the unchanged PDF.</p>
<p>Items marked as definitions, hypotheses, identities, or specializations have the precise roles
described in their cards. Equation (18) is mapped through a hinge representation; the map does
not claim a separate theorem reproducing its displayed equality verbatim.</p>
<p><a href="../README.txt">Setup and verification</a> ·
<a href="PROOF_MAP.txt">Complete plain-text guide</a> ·
<a href="proof_map.tsv">Download TSV</a> ·
<a href="DECLARATIONS.html">All declaration names</a> ·
<a href="../manuscript/junpei_version.pdf">Supplied PDF</a></p>
<p>PDF links use the supplied file's page numbers. Source links open a raw <code>.lean</code>
file; use the displayed line number in your Lean editor. A browser may download source files.</p>
</div>
<div class="controls">
<div class="fields">
<div class="field search"><label for="search">Search paper items</label>
<input id="search" type="search" placeholder="e.g. Theorem 2, Equation (19), initial risk" autocomplete="off"></div>
<div class="field"><label for="status">Status</label>
<select id="status"><option value="">All statuses</option>__STATUSES__</select></div>
<div class="field"><label for="kind">Item type</label>
<select id="kind"><option value="">All item types</option>__KINDS__</select></div>
</div>
<p id="count" role="status" aria-live="polite">__TOTAL__ of __TOTAL__ paper items shown</p>
</div>
<noscript><p>JavaScript is disabled. Every item is shown; use your browser's Find command.</p></noscript>
<main>__CARDS__</main>
<script>
"use strict";
const cards = Array.from(document.querySelectorAll("article.item"));
const texts = cards.map(card => card.textContent.toLocaleLowerCase());
const search = document.getElementById("search");
const status = document.getElementById("status");
const kind = document.getElementById("kind");
const count = document.getElementById("count");
function filter() {
  const words = search.value.toLocaleLowerCase().trim().split(/\s+/).filter(Boolean);
  let shown = 0;
  cards.forEach((card, index) => {
    const matches = (!status.value || card.dataset.status === status.value)
      && (!kind.value || card.dataset.kind === kind.value)
      && words.every(word => texts[index].includes(word));
    card.hidden = !matches;
    if (matches) shown++;
  });
  count.textContent = `${shown} of ${cards.length} paper items shown`;
}
search.addEventListener("input", filter);
status.addEventListener("change", filter);
kind.addEventListener("change", filter);
</script>
</body></html>
'''
    for key, value in {"__TOTAL__": str(len(items)), "__REFERENCES__": str(references),
                       "__STATUSES__": options("status"), "__KINDS__": options("kind"),
                       "__CARDS__": "\n".join(cards)}.items():
        document = document.replace(key, value)
    (root / "docs" / "PROOF_MAP.html").write_text(document, encoding="utf-8")
    return len(items)


def build(root: Path) -> tuple[int, int, int]:
    root = root.resolve()
    inventory = root / "declarations.json"
    expected = json.loads(inventory.read_text(encoding="utf-8-sig"))
    if not isinstance(expected, list) or not all(isinstance(name, str) for name in expected):
        raise ValueError("declarations.json must be a list of fully qualified Lean names")
    if len(expected) != len(set(expected)):
        raise ValueError("declarations.json contains duplicate names")
    files = sorted((root / "Draft0").rglob("*.lean")) + sorted((root / "Junpei").rglob("*.lean"))
    if not files:
        raise ValueError("No Lean modules found under Draft0/ and Junpei/")
    modules = [scan_module(path, root) for path in files]
    declarations = sorted((d for m in modules for d in m.declarations), key=lambda d: d.name)
    names = [declaration.name for declaration in declarations]
    if len(names) != len(set(names)):
        raise ValueError("Scanned public declarations contain duplicate fully qualified names")
    if set(names) != set(expected):
        missing = sorted(set(expected) - set(names))
        extra = sorted(set(names) - set(expected))
        raise ValueError("Index differs from declarations.json: " + json.dumps(
            {"missing_from_scan": missing, "extra_in_scan": extra}, ensure_ascii=False
        ))
    docs = root / "docs"
    docs.mkdir(exist_ok=True)
    write_tsv(docs / "declarations.tsv", declarations)
    write_modules(docs / "MODULES.txt", modules)
    write_html(docs / "DECLARATIONS.html", declarations, modules)
    paper_items = write_proof_map(root)
    return len(declarations), len(modules), paper_items


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="Lean project root (default: this script's parent project)")
    args = parser.parse_args()
    try:
        declarations, modules, paper_items = build(args.root)
    except (OSError, ValueError) as error:
        print(f"Index generation failed: {error}", file=sys.stderr)
        return 1
    print(f"Indexed {declarations:,} declarations in {modules} modules; names match declarations.json.")
    print("Wrote docs/declarations.tsv, docs/DECLARATIONS.html, and docs/MODULES.txt.")
    print(f"Wrote docs/PROOF_MAP.html from {paper_items} paper items in docs/proof_map.tsv.")
    print("Navigation only: Lean proof verification is performed separately by verify.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
