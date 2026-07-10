# Deploy Lattice Mini to Hugging Face

The `space/` folder holds **deploy source only**. Run `setup_from_repo.sh` to copy
model code, tokenizer, and checkpoint before pushing to HF.

## One-time setup

```bash
cd space
./setup_from_repo.sh
hf auth login --add-to-git-credential
```

## Push to Hugging Face (not GitHub!)

```bash
git clone https://huggingface.co/spaces/oli-mebberson/lattice-mini /tmp/lattice-mini
cd /tmp/lattice-mini
git remote -v   # must show huggingface.co, NOT github.com

cp -r /path/to/nano-gpt/space/* .
cp /path/to/nano-gpt/space/.gitattributes .

git lfs install
git lfs track "*.pt"
git add .
git commit -m "Update Lattice Mini"
git push
```

If push asks for a password, use your [HF token](https://huggingface.co/settings/tokens)
(write access), not your account password.

## Hardware

This Space uses **ZeroGPU** (free). `app.py` includes `@spaces.GPU` for that tier.
To use CPU basic instead, create a **new** Space and select CPU basic at creation time.
