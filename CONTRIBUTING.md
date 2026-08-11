# Contributing and CI setup

This file documents how to configure CI/CD and local development to work with DVC and Backblaze B2.

## GitHub repository secrets (required for CI/CD)

Add the following secrets in your repository Settings → Secrets → Actions:

- `B2_KEY_ID` — Backblaze S3-compatible application Key ID (use the app key ID)
- `B2_APP_KEY` — Backblaze application Key (the secret)
- `B2_REGION` — Backblaze region code (example: `us-west-000`, `eu-central-003`)

These secrets are used in the GitHub Actions workflows to run `dvc pull` before tests and deployments.

## Local development

1. Create and activate the virtualenv (repo root):

```bash
python3 -m venv .venv.nosync
source .venv.nosync/bin/activate
```

2. Install dependencies and DVC with S3 support:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m pip install "dvc[s3]"
```

3. Export your Backblaze credentials in the shell before running training/tests (do NOT commit these):

```bash
export AWS_ACCESS_KEY_ID="YOUR_B2_KEY_ID"
export AWS_SECRET_ACCESS_KEY="YOUR_B2_APP_KEY"
export AWS_DEFAULT_REGION="your-region-code"
```

4. Use the helper script or Makefile to run training:

```bash
# using Makefile
make install
make train

# or using helper directly
./scripts/run_train.sh --train-csv data/train.csv
```

5. To skip automatic `dvc pull` (for offline runs or CI tests that mock data), set:

```bash
export SKIP_DVC_PULL=1
```

If you need help creating an app key in Backblaze, see Backblaze's documentation for Application Keys and S3-compatible API.
