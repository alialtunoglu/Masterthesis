# Skill: Python Project Development

When writing Python project code:

1. Use pathlib.Path for file paths.
2. Use argparse for command-line scripts.
3. Use pandas for CSV outputs.
4. Use json for split/config metadata.
5. Use tqdm for long loops.
6. Use type hints.
7. Keep functions small and composable.
8. Write reusable helper functions in src/utils.
9. Avoid hardcoded absolute paths.
10. Assume scripts are run from the project root.

Every executable script should have:

if __name__ == "__main__":
    main()
