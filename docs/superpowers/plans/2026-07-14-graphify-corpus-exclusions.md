# Graphify Corpus Exclusions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the repository Graphify outputs without repository-root `datasets/` data or `results/` artifacts while retaining all source code, including `src/datasets/`.

**Architecture:** Add anchored, Graphify-specific ignore rules at the repository root. Prove the detection boundary before rebuilding, then use Graphify's forced incremental update to prune now-ignored sources and regenerate the graph, report, and HTML without touching the excluded files.

**Tech Stack:** Graphify 0.9.9, `.graphifyignore`, PowerShell, Python JSON validation

## Global Constraints

- Exclude only repository-root `datasets/` and `results/` trees.
- Retain `src/datasets/` and every other source-code directory.
- Do not delete, move, or modify dataset or result artifacts.
- Preserve the currently valid Graphify outputs until replacement output generation succeeds.
- Git author identity is currently unset; do not modify Git identity and do not claim a commit was created.

---

### Task 1: Add and verify persistent corpus exclusions

**Files:**
- Create: `.graphifyignore`
- Reference: `.gitignore`
- Test output: `graphify-out/.exclusion_detect.json`

**Interfaces:**
- Consumes: Graphify's gitignore-compatible detection rules.
- Produces: Anchored `/datasets/` and `/results/` exclusions used by every later Graphify scan.

- [ ] **Step 1: Run the pre-change detection check**

Run:

```powershell
$py = Get-Content -Raw 'graphify-out\.graphify_python'
& $py -c "from pathlib import Path; from graphify.detect import detect; d=detect(Path('.')); files=[p for xs in d['files'].values() for p in xs]; results=[p for p in files if Path(p).resolve().is_relative_to((Path('results')).resolve())]; print('results_included=',len(results)); raise SystemExit(0 if results else 1)"
```

Expected: exit `0` and `results_included` greater than zero, proving the check detects the current unwanted corpus content.

- [ ] **Step 2: Create the minimal ignore file**

Create `.graphifyignore` with exactly:

```gitignore
/datasets/
/results/
```

- [ ] **Step 3: Run filtered detection and persist its result**

Run:

```powershell
$py = Get-Content -Raw 'graphify-out\.graphify_python'
& $py -c "import json; from pathlib import Path; from graphify.detect import detect; d=detect(Path('.')); Path('graphify-out/.exclusion_detect.json').write_text(json.dumps(d,ensure_ascii=False),encoding='utf-8'); files=[Path(p).resolve() for xs in d['files'].values() for p in xs]; root=Path('.').resolve(); bad=[p for p in files if p.is_relative_to(root/'datasets') or p.is_relative_to(root/'results')]; src=[p for p in files if p.is_relative_to(root/'src'/'datasets')]; print('excluded_matches=',len(bad)); print('src_dataset_code=',len(src)); raise SystemExit(1 if bad else 0)"
```

Expected: exit `0`, `excluded_matches= 0`; `src_dataset_code` must be greater than zero when supported files exist under `src/datasets/`.

- [ ] **Step 4: Check configuration formatting**

Run:

```powershell
git diff --check -- .graphifyignore
```

Expected: exit `0` with no formatting errors.

### Task 2: Rebuild the filtered graph safely

**Files:**
- Preserve until success: `graphify-out/graph.json`
- Preserve until success: `graphify-out/GRAPH_REPORT.md`
- Preserve until success: `graphify-out/graph.html`
- Create temporarily: `C:\tmp\masterthesis-graphify-pre-exclusion\`
- Regenerate: `graphify-out/graph.json`
- Regenerate: `graphify-out/GRAPH_REPORT.md`
- Regenerate: `graphify-out/graph.html`

**Interfaces:**
- Consumes: `.graphifyignore` and the existing Graphify manifest/graph.
- Produces: A smaller graph whose ignored sources have been pruned and whose communities/report/HTML match the filtered corpus.

- [ ] **Step 1: Back up the three valid outputs**

Run:

```powershell
$backup = 'C:\tmp\masterthesis-graphify-pre-exclusion'
New-Item -ItemType Directory -Force -Path $backup | Out-Null
Copy-Item -LiteralPath 'graphify-out\graph.json','graphify-out\GRAPH_REPORT.md','graphify-out\graph.html' -Destination $backup -Force
```

Expected: the backup directory contains all three non-empty files.

- [ ] **Step 2: Run the intentional shrinking update**

Run:

```powershell
graphify update . --force
```

Expected: exit `0`, an updated code graph, and regenerated `graphify-out/graph.json`, `graphify-out/GRAPH_REPORT.md`, and `graphify-out/graph.html`. The `--force` flag explicitly permits removal of ignored result nodes.

- [ ] **Step 3: Restore only if generation failed**

If Step 2 exits nonzero or any required output is missing/empty, run:

```powershell
$backup = 'C:\tmp\masterthesis-graphify-pre-exclusion'
Copy-Item -LiteralPath "$backup\graph.json","$backup\GRAPH_REPORT.md","$backup\graph.html" -Destination 'graphify-out' -Force
```

Expected: the previously valid outputs are restored. Do not run this step after a successful update.

### Task 3: Verify excluded paths and final artifacts

**Files:**
- Verify: `graphify-out/graph.json`
- Verify: `graphify-out/GRAPH_REPORT.md`
- Verify: `graphify-out/graph.html`
- Remove after success: `graphify-out/.exclusion_detect.json`
- Remove after success: `C:\tmp\masterthesis-graphify-pre-exclusion\`

**Interfaces:**
- Consumes: the rebuilt Graphify artifacts.
- Produces: Evidence that excluded artifacts are absent and source code remains queryable.

- [ ] **Step 1: Validate graph structure and source paths**

Run:

```powershell
$g = Get-Content -Raw 'graphify-out\graph.json' | ConvertFrom-Json
$nodes = @($g.nodes)
$edges = if ($null -ne $g.links) { @($g.links) } else { @($g.edges) }
$bad = @($nodes | Where-Object { $_.source_file -match '(^|[\\/])(datasets|results)[\\/]' })
$srcDataset = @($nodes | Where-Object { $_.source_file -match '(^|[\\/])src[\\/]datasets[\\/]' })
Write-Output "nodes=$($nodes.Count) edges=$($edges.Count) excluded_nodes=$($bad.Count) src_dataset_nodes=$($srcDataset.Count)"
if ($nodes.Count -eq 0 -or $edges.Count -eq 0 -or $bad.Count -ne 0) { exit 1 }
```

Expected: exit `0`, positive node/edge counts, and `excluded_nodes=0`. `src_dataset_nodes` must be positive when `src/datasets/` contains extractable symbols.

- [ ] **Step 2: Validate report and HTML outputs**

Run:

```powershell
$report = Get-Content -Raw 'graphify-out\GRAPH_REPORT.md'
$html = Get-Content -Raw 'graphify-out\graph.html'
$ok = $report -match '(?m)^## God Nodes' -and $report -match '(?m)^## Surprising Connections' -and $report -match '(?m)^## Suggested Questions' -and $html.Length -gt 1000
if (-not $ok) { exit 1 }
```

Expected: exit `0`; all mandatory report sections and a non-empty interactive HTML artifact exist.

- [ ] **Step 3: Run Graphify's graph-health diagnostic**

Run:

```powershell
graphify diagnose multigraph --graph graphify-out\graph.json
```

Expected: exit `0`. Record and surface any dangling/collapsed-edge warnings without hiding them.

- [ ] **Step 4: Remove only temporary verification artifacts**

After all verification steps pass, resolve both paths and confirm they are exactly `graphify-out/.exclusion_detect.json` under the repository and `C:\tmp\masterthesis-graphify-pre-exclusion` before removal. Then run:

```powershell
Remove-Item -LiteralPath 'graphify-out\.exclusion_detect.json' -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath 'C:\tmp\masterthesis-graphify-pre-exclusion' -Recurse -Force
```

Expected: temporary detection and backup artifacts are gone; `.graphifyignore` and the three final outputs remain.

- [ ] **Step 5: Report the uncommitted configuration change**

Run:

```powershell
git status --short -- .graphifyignore docs/superpowers/specs/2026-07-14-graphify-corpus-exclusions-design.md docs/superpowers/plans/2026-07-14-graphify-corpus-exclusions.md graphify-out
```

Expected: report the exact status. Do not configure Git identity or claim a commit, because the repository currently has no author identity configured.

