# Graphify Corpus Exclusions Design

## Goal

Rebuild the repository knowledge graph without raw datasets or generated experiment results, while retaining all source code, including dataset-related implementation under `src/datasets/`.

## Scope

- Exclude the repository-root `datasets/` directory and everything beneath it.
- Exclude the repository-root `results/` directory and everything beneath it.
- Retain `src/datasets/` and every other source-code directory.
- Do not delete, move, or modify any dataset or result artifact.
- Regenerate `graphify-out/graph.json`, `graphify-out/GRAPH_REPORT.md`, and `graphify-out/graph.html`.

## Design

Add a repository-root `.graphifyignore` containing anchored rules:

```gitignore
/datasets/
/results/
```

Graphify merges `.gitignore` and `.graphifyignore`, with `.graphifyignore` providing Graphify-specific exclusions. Anchored root rules prevent accidental exclusion of code directories such as `src/datasets/`.

Perform a fresh corpus detection and full graph rebuild. Because the filtered graph is intentionally smaller than the existing graph, allow the intentional shrink only for this rebuild. Preserve the existing final outputs until the replacement graph has been built successfully.

## Data Flow

1. Graphify reads the repository root and both ignore files.
2. Detection omits `datasets/**` and `results/**` while retaining source code.
3. Structural and eligible semantic extraction run on the filtered corpus.
4. Graphify rebuilds communities, labels, report, JSON, and HTML.
5. Verification checks both included and excluded path invariants before accepting the outputs.

## Failure Handling

- If detection includes a file beneath either excluded root, stop before overwriting final outputs.
- If `src/datasets/` source files are absent when they exist on disk, stop and correct the ignore rules.
- If extraction produces an empty graph or output generation fails, keep the previously valid final outputs.
- Surface graph-health warnings rather than hiding them.

## Verification

- Detection contains zero files beneath repository-root `datasets/` or `results/`.
- Detection retains supported files beneath `src/datasets/` when present.
- No node in `graph.json` has `source_file` beneath repository-root `datasets/` or `results/`.
- At least one code node is present and `graph.json` parses successfully.
- `GRAPH_REPORT.md` contains God Nodes, Surprising Connections, and Suggested Questions.
- `graph.html` exists and is non-empty.

