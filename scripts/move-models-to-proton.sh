#!/usr/bin/env bash
# Move Lattice Models from Google Drive -> Proton Drive (via local staging).
# Run AFTER Google account has free storage (over-quota blocks downloads).

set -euo pipefail

GDRIVE="$HOME/Library/CloudStorage/GoogleDrive-olimebb@gmail.com/My Drive/Lattice Models"
PROTON="$HOME/Library/CloudStorage/ProtonDrive-oli@mebberson.com-folder/Lattice Models"
STAGE="$HOME/Downloads/lattice-models-staging"

if [[ ! -d "$GDRIVE" ]]; then
  echo "Google Drive folder not found: $GDRIVE"
  exit 1
fi
if [[ ! -d "$(dirname "$PROTON")" ]]; then
  echo "Proton Drive not mounted. Install/sign in to Proton Drive first."
  exit 1
fi

mkdir -p "$STAGE" "$PROTON"

echo "==> Staging from Google Drive..."
rm -rf "$STAGE"
ditto "$GDRIVE" "$STAGE"

echo "==> Uploading to Proton Drive..."
for dir in Air Mini Pulse; do
  if [[ -d "$STAGE/$dir" ]]; then
    echo "   $dir"
    ditto "$STAGE/$dir" "$PROTON/$dir"
  fi
done

echo "==> Done."
du -sh "$PROTON"/* 2>/dev/null || true
