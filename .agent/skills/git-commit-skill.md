# Skill: Git Commit Support

When asked to prepare or suggest a Git commit:

1. Run or request git status.
2. Summarize modified, added, and deleted files.
3. Identify files that should not be committed.
4. Check whether .gitignore needs updates.
5. Suggest a clean commit message.

Commit message examples:

feat(datasets): add dataset analysis and split generation
docs(datasets): document dataset layout and split policy
fix(splits): preserve class distribution for small classes
refactor(utils): add reusable JSON and CSV helpers
chore(git): ignore checkpoints and generated caches

Never suggest committing:

- raw datasets
- checkpoints
- large logs
- Python cache files
- virtual environments
