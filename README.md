# FF Decomposition Experiments

This repository contains the minimum code needed to reproduce the paper's MLP,
CNN, and MLP-Mixer experiments. Run every command from the repository root on a
Linux machine with one CUDA GPU.

## Setup

```bash
conda env create -f environment.yml
conda activate ff-decomposition
python DatasetBootstrap.py --data-dir "$PWD/data"
```

The bootstrap command prepares MNIST, FashionMNIST, CIFAR-10, and CIFAR-100.
The MLP-Mixer scheduler prepares PathMNIST and TinyImageNet when required.

## MLP experiments

First select the goodness threshold from 1, 2, and 4 using constant Adam:

```bash
T="$PWD/runs/mlp-thresholds"
python MLPThresholdSweepScheduler.py plan \
  --run-dir "$T" --project-root "$PWD" --python "$(command -v python)"
nohup python MLPThresholdSweepScheduler.py launch --run-dir "$T" \
  > "$T/launcher.log" 2>&1 &
python MLPThresholdSweepScheduler.py status --run-dir "$T"
```

Repeat the status command until it reports `training_complete`, then freeze the
validation-selected thresholds:

```bash
python MLPThresholdSweepScheduler.py select --run-dir "$T"
```

Run the optimizer sweep over constant Adam, cosine-annealed Adam, and
step-decayed SGD:

```bash
M="$PWD/runs/mlp"
python MLPOptimizerSweepScheduler.py plan \
  --run-dir "$M" --project-root "$PWD" --python "$(command -v python)" \
  --thresholds-json "$T/selected_thresholds.json"
nohup python MLPOptimizerSweepScheduler.py launch --run-dir "$M" \
  > "$M/launcher.log" 2>&1 &
python MLPOptimizerSweepScheduler.py status --run-dir "$M"
```

After all candidates finish, select by validation accuracy, evaluate the chosen
checkpoints once on the test set, and write `runs/mlp/summary.csv`:

```bash
python MLPOptimizerSweepScheduler.py finalize \
  --run-dir "$M" --data-dir "$PWD/data" --device cuda
python MLPOptimizerSweepScheduler.py aggregate --run-dir "$M"
```

## CNN experiments

This compares cosine-annealed Adam and step-decayed SGD for CE (Local,
Multi-Head), CE (Global, Multi-Head), and CE (Global, Terminal).

```bash
C="$PWD/runs/cnn"
mkdir -p "$C"
nohup python CNNOptimizerSweepScheduler.py \
  --run-dir "$C" --project-root "$PWD" --python "$(command -v python)" \
  > "$C/scheduler.log" 2>&1 &
cat "$C/status.json"
```

The final table is `runs/cnn/cnn_paper_results.csv`.

## MLP-Mixer experiments

This runs the same three cross-entropy variants with the fixed D5-W256 AdamW
configuration on CIFAR-10, CIFAR-100, PathMNIST, and TinyImageNet.

```bash
X="$PWD/runs/mixer"
mkdir -p "$X"
nohup python ReducedMixerBenchmarkScheduler.py \
  --run-dir "$X" --project-root "$PWD" --python "$(command -v python)" \
  > "$X/scheduler.log" 2>&1 &
cat "$X/status.json"
```

The final table is `runs/mixer/reduced_mixer_summary.csv`.

Rerunning a scheduler with the same run directory resumes only outputs that
pass its configuration, source, checkpoint, and split-integrity checks.
