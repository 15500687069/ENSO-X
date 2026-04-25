# Upload ENSO-X To GitHub

This package is prepared as a standalone Git repository candidate.

## Recommended Repository Name

- `ENSO-X`

## Suggested Steps

1. Create an empty repository under your GitHub account:

```bash
https://github.com/15500687069/ENSO-X
```

2. Set your local Git identity if needed:

```bash
git config user.name "15500687069"
git config user.email "your-email@example.com"
```

3. Initialize and push:

```bash
git init -b main
git add .
git commit -m "init ENSO-X release package"
git remote add origin https://github.com/15500687069/ENSO-X.git
git push -u origin main
```

## Notes

- Large checkpoint binaries are not stored directly inside this repository.
- The kept checkpoint aliases and server-side paths are documented in `checkpoints/MANIFEST.json`.
- If you prefer SSH, replace the remote URL with:

```bash
git@github.com:15500687069/ENSO-X.git
```
