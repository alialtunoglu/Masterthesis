# Git Rules

Follow disciplined Git practices.

Repository hygiene:

1. Do not commit raw datasets.
2. Do not commit large model checkpoints.
3. Do not commit temporary files, caches, or logs unless explicitly requested.
4. Add or update .gitignore when new generated directories appear.
5. Keep source code, configs, documentation, and small metadata files versioned.
6. Split JSON files may be committed because they are small and important for reproducibility.
7. Results CSV files may be committed only if they are small and thesis-relevant.

Commit style:

Use clear commit messages.

Preferred format:

type(scope): short description

Examples:

feat(datasets): add deterministic split generation
fix(plantpathology): filter multi-label samples correctly
docs(datasets): document dataset layout and split policy
refactor(training): extract reusable training loop
chore(git): update gitignore for checkpoints

Allowed types:

- feat
- fix
- docs
- refactor
- test
- chore
- perf

Before suggesting a commit:

1. Run relevant scripts if possible.
2. Check git status.
3. Summarize changed files.
4. Suggest a meaningful commit message.
