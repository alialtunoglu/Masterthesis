# Portable Notebook Dataset Bootstrap Design

## Goal

Generated experiment notebooks must prepare supported datasets safely in Kaggle, Google Colab, and local Jupyter environments without embedding credentials or repeating downloads.

## Scope

The change is limited to the shared portable-notebook generator. AppleLeaf9 and PlantVillage keep their pinned GitHub checkout behavior. Plant Pathology 2021 gains environment-aware setup.

## Plant Pathology bootstrap flow

The generated notebook will follow this order:

1. If `datasets/PlantPathology2021/train.csv` and `train_images/` already exist, reuse them.
2. If `/kaggle/input/plant-pathology-2021-fgvc8` exists, link its `train.csv` and `train_images/` into the project dataset directory without copying the image corpus.
3. Otherwise, install the Kaggle CLI only when unavailable and inspect supported authentication sources.
4. In Google Colab, when authentication is absent, request `kaggle.json` through the temporary file-upload widget, place it under `~/.kaggle/`, and restrict its permissions.
5. In local Jupyter, accept existing `KAGGLE_API_TOKEN`, `~/.kaggle/access_token`, or `~/.kaggle/kaggle.json`; if none exists, stop with actionable instructions.
6. Download and extract the competition archive, then validate that the required metadata and image directory exist.

The notebook will never include, print, commit, or export credential contents.

## Error handling

The generated cell will expose Kaggle command output and translate common failures into actionable messages:

- missing authentication: configure a token or upload `kaggle.json` in Colab;
- authorization failure: accept the competition rules using the same Kaggle account;
- unavailable Kaggle input: add the competition as a Kaggle Notebook input;
- missing final files: report the expected paths and stop before training.

Unknown command failures will preserve the original output and return code.

## Code structure

The existing `app.portable_notebook._dataset_setup()` remains the single dispatch point. Small source-generation helpers may be introduced only where they make the emitted Plant Pathology cell testable; no platform abstraction or new dependency is needed.

## Tests

Tests will inspect and execute the generated setup source with temporary paths and mocked environment boundaries. They will cover:

- Kaggle mounted-input reuse without invoking the CLI;
- Colab upload path when credentials are absent;
- existing credentials bypassing upload;
- already-prepared datasets bypassing all acquisition;
- clear failure when local credentials are absent;
- preservation of AppleLeaf9 and PlantVillage pinned sources;
- valid notebook JSON generation.

No dataset download or training will run during tests.

## Success criteria

A newly exported Plant Pathology notebook can reach a validated dataset layout in Kaggle, Colab, or local Jupyter with at most one credential upload or competition-input selection by the user. Secrets are never serialized into the notebook or its generated artifacts.
