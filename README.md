# FF Decomposition

Code and reproducibility material for **Disentangling Objective Design and
Gradient Locality in Supervised Forward--Forward Learning** by James Gong and
Waleed Abdulla.

The study separates three choices that are often bundled together in
supervised Forward--Forward (FF) learning: the supervision objective, the
placement of that objective, and whether gradients cross layer boundaries. It
contains the eleven MLP variants reported in the paper and cross-entropy
reference suites for CNNs and MLP-Mixers.

## What is included

- **MLP decomposition:** Vanilla FF, global and terminal FF variants,
  normalization ablations, full-comparison FF variants, and three
  cross-entropy references on MNIST, FashionMNIST, CIFAR-10, and CIFAR-100.
- **CNN references:** local multi-head, global multi-head, and global terminal
  cross-entropy on the same four datasets.
- **MLP-Mixer references:** the same three cross-entropy variants on CIFAR-10,
  CIFAR-100, PathMNIST, and TinyImageNet.
- **Validation-safe sweeps:** optimizer candidates defer the test split;
  selection uses validation accuracy across seeds 424, 425, and 426; only the
  selected checkpoints are evaluated on test.
- **Auditable artifacts:** each scheduler records its commands, source hashes,
  configuration, environment, checkpoints, logs, state, and result summaries.

The exact method names and settings are fixed in
[`docs/PAPER_PROTOCOL.md`](docs/PAPER_PROTOCOL.md) and the machine-readable
[`configs/paper_protocol.json`](configs/paper_protocol.json).

## Installation

The final GPU runs used Python 3.10, PyTorch 2.6.0, torchvision 0.21.0, CUDA
12.4, and MedMNIST 3.0.2. A reference Conda environment matching this core
stack is:

```bash
conda env create -f environment.yml
conda activate ff-decomposition
python -m pip install -e .
python -m pip check
```

For another CUDA version, create a Python 3.10 environment, install the
matching official PyTorch build, and then run:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Run all paper experiments on a CUDA GPU. CPU execution is useful for unit
tests and small smoke tests, but is not a practical way to reproduce the full
matrix.

Download the four torchvision datasets required by the MLP and CNN schedulers
before planning or launching either sweep:

```bash
python DatasetBootstrap.py --data-dir "$PWD/data"
```

## Paper experiment entry points

### MLP threshold sweep

The preliminary sweep contains 144 constant-Adam, validation-only candidates:
four thresholded FF methods, four datasets, three thresholds, and three seeds.
It produces the frozen threshold mapping consumed by the optimizer sweep.

```bash
THRESHOLD_DIR="$PWD/runs/mlp-threshold-sweep"
python MLPThresholdSweepScheduler.py plan \
  --run-dir "$THRESHOLD_DIR" \
  --project-root "$PWD" \
  --python "$(command -v python)"
nohup python MLPThresholdSweepScheduler.py launch \
  --run-dir "$THRESHOLD_DIR" \
  > "$THRESHOLD_DIR/launcher.log" 2>&1 &
echo $! > "$THRESHOLD_DIR/launcher.pid"
python MLPThresholdSweepScheduler.py status --run-dir "$THRESHOLD_DIR"
# Repeat status checks; run select only after it reports training_complete.
python MLPThresholdSweepScheduler.py select --run-dir "$THRESHOLD_DIR"
```

### MLP optimizer sweep

Planning freezes source hashes, the 360 validation-only candidate jobs, and
the thresholds selected by the separate threshold study. Planning does not
train a model.

```bash
RUN_DIR="$PWD/runs/mlp-paper-sweep"
python MLPOptimizerSweepScheduler.py plan \
  --run-dir "$RUN_DIR" \
  --project-root "$PWD" \
  --python "$(command -v python)" \
  --thresholds-json "$THRESHOLD_DIR/selected_thresholds.json"
```

Launch or resume the candidates as a detached process:

```bash
nohup python MLPOptimizerSweepScheduler.py launch \
  --run-dir "$RUN_DIR" \
  > "$RUN_DIR/launcher.log" 2>&1 &
echo $! > "$RUN_DIR/launcher.pid"
```

Then inspect, select, test, and aggregate:

```bash
python MLPOptimizerSweepScheduler.py status --run-dir "$RUN_DIR"
python MLPOptimizerSweepScheduler.py finalize \
  --run-dir "$RUN_DIR" --data-dir "$PWD/data" --device cuda
python MLPOptimizerSweepScheduler.py aggregate --run-dir "$RUN_DIR"
```

An optional UTC deadline can be added to `launch`, for example
`--stop-before-utc 2026-10-05T06:00:00Z`. The scheduler stops starting new
candidates before the deadline and can be resumed with the same command.

### CNN optimizer sweep

The CNN scheduler compares cosine-annealed Adam and step-decayed SGD for all
three cross-entropy variants. Its candidate runs also defer test evaluation,
select a profile by validation accuracy, and test only the selected
checkpoints.

```bash
RUN_DIR="$PWD/runs/cnn-paper-sweep"
mkdir -p "$RUN_DIR"
nohup python CNNOptimizerSweepScheduler.py \
  --run-dir "$RUN_DIR" \
  --project-root "$PWD" \
  --python "$(command -v python)" \
  > "$RUN_DIR/scheduler.log" 2>&1 &
echo $! > "$RUN_DIR/scheduler.pid"
cat "$RUN_DIR/status.json"
```

The scheduler resumes verified candidates from the same run directory, freezes
the validation winner for each method and dataset, evaluates the selected
checkpoints once, and writes the final summaries. An optional deadline can be
supplied with `--stop-before-utc` in the same ISO-8601 form used by the MLP
scheduler.

### MLP-Mixer references

The MLP-Mixer suite has one fixed AdamW profile and 36 runs. The scheduler is
restart-safe and skips only outputs that pass its provenance checks.

```bash
RUN_DIR="$PWD/runs/mixer-paper-baselines"
mkdir -p "$RUN_DIR"
nohup python ReducedMixerBenchmarkScheduler.py \
  --run-dir "$RUN_DIR" \
  --project-root "$PWD" \
  --python "$(command -v python)" \
  > "$RUN_DIR/scheduler.log" 2>&1 &
echo $! > "$RUN_DIR/scheduler.pid"
```

TinyImageNet is downloaded and structurally validated when it is not already
available. PathMNIST is provided through MedMNIST. The general Mixer runner
supports more datasets, including COIL-100 and additional MedMNIST tasks, but
they are outside the active paper table.

## Output and reported results

Generated datasets and run directories are intentionally ignored by Git. Copy
complete run directories off the compute server before releasing resources;
the manifest, source hashes, logs, checkpoints, state markers, and summaries
are needed to audit a run. Manifests record machine and absolute-path metadata,
so review an archive before publishing it outside the research group.

[`paper_results/reported_accuracy.csv`](paper_results/reported_accuracy.csv)
contains the 68 aggregate rows in the paper-table snapshot used to prepare this
release: 64 reported results and four intentional CIFAR-100 omissions for the
full-comparison FF family. It is not a substitute for per-seed run artifacts,
and it must be reconciled with the final manuscript before tagging a release.
The sanitized provenance export verifies 50 of the
64 reported rows against complete local three-seed artifacts and lists the
remaining 14 separately in
[`paper_results/unsupported_reported_rows.csv`](paper_results/unsupported_reported_rows.csv).
The evidence sources and matching procedure are documented in
[`paper_results/PROVENANCE.md`](paper_results/PROVENANCE.md).

## Tests

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check .
```

The tests use synthetic data and inspect protocol construction; they do not
download datasets or start the full training matrix.

Older diagnostic schedulers are retained so earlier runs remain inspectable,
but they are not installed as command-line entry points and must not be used as
the current paper matrix. The four paper launchers documented above are the
authoritative orchestration paths.

## Interpretation

The paper-facing local variants use independent layer or block optimizers,
whereas their global multi-head counterparts jointly optimize their losses
with one optimizer. These rows therefore compare the complete local and global
training formulations used in the paper. They should not be described as a
single detach-only intervention. The presented `ff` and `ff-matched-ge` rows
also use independently validation-selected thresholds. Compatibility controls
used to test gradient routing in isolation remain available in the MLP runner
but are excluded from the eleven-row paper matrix.

## Citation and release status

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). Before a
public tagged release, complete the short
[`docs/RELEASE_CHECKLIST.md`](docs/RELEASE_CHECKLIST.md), including selection
of a software license and insertion of the article DOI when available.
