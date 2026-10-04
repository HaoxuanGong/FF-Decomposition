# Result provenance

This directory separates manuscript-reported aggregates from the local artifacts that can presently substantiate them. `reported_accuracy.csv` is the manuscript table transcription. `per_seed_provenance.csv` contains only rows for which a complete three-seed local result group reproduces the reported mean and sample standard deviation after rounding to two decimal places. `unsupported_reported_rows.csv` identifies reported rows for which no such complete local artifact was found.

## Matching method

The audit grouped test results by architecture, dataset, method, selected optimizer profile, and (where applicable) goodness threshold. A group was accepted only when it contained seeds 424, 425, and 426. The mean and sample standard deviation (denominator `n-1`) were recomputed from those three test accuracies. Both values had to equal the corresponding entry in `reported_accuracy.csv` after rounding to two decimal places.

The audit found exact support for 50 of the 64 reported aggregate rows, represented by 150 per-seed records:

| Source run group | Supported aggregates |
|---|---:|
| MLP optimizer investigation | 21 |
| MLP threshold sweep | 6 |
| NeSI missing-variant study | 3 |
| CNN optimizer sweep | 12 |
| Verified reduced MLP-Mixer benchmark | 8 |

All 50 accepted groups passed the recomputation check. The remaining 14 rows comprise 10 MLP rows and the four MLP-Mixer `ce-matched-ge` rows.

## Local evidence used

The following paths are relative to the workspace that was audited. They are recorded here to make the assembly traceable; the large raw result directories are intentionally not copied into the source repository.

- `mlp_optimizer_investigation_20260913/completed_20260914/analysis/<dataset>/selected.csv` supplies selected profiles and per-seed test accuracies. The adjacent `selection.json` files supply checkpoint, split, run, configuration, history, and source hashes. `mlp_optimizer_investigation_20260913/completed_20260914/manifest.json` records Git revision `aa3b67c1725848c847cda717a1cf5a81d7e42890`, the three optimizer profiles, and the validation-only selection policy.
- `ff_threshold_sweep_20260914/final_records/results/<threshold>/<dataset>/<method>/seed_<seed>/test.json` supplies the accepted threshold-sweep test results. The adjacent `analysis/<dataset>/selection.json` files supply validation choices and checkpoint/split hashes. `final_records/manifest.json` records Git revision `aa3b67c1725848c847cda717a1cf5a81d7e42890` and the source hashes.
- `nesi_ff11_results_9299983/study` supplies the accepted NN-FF rows for MNIST and CIFAR-10 and the FC-FF global multi-head row for CIFAR-10. Its per-seed `run.json`/`test.json` files and analysis selections record the checkpoint and split hashes. The archive does not explicitly record a Git revision, so that field is left blank.
- `cnn_optimizer_results_20261001T090413Z/finalization/summary.json` supplies all CNN per-seed selected-test records. `selection.json` records validation-only profile selection plus checkpoint, source, and split hashes.
- `paper_revision_source_20260828/verified_reduced_mixer_20260828_120900/reduced_mixer_summary.csv` supplies the MLP-Mixer local multi-head and global terminal per-seed values. Its `manifest.json` and `source_snapshot.sha256` record the exact runner snapshot. That compact archive does not contain per-seed checkpoint or split hashes, so those fields are blank.

## Field interpretation

- `selected_profile` uses the paper-facing optimizer profile name. `adamw_cosine_fixed` denotes the fixed MLP-Mixer recipe rather than a profile selected from a sweep.
- `threshold` is populated only for thresholded FF runs.
- `source_revision` is populated only when the artifact explicitly records a Git revision. A blank value is not an inferred revision.
- `checkpoint_sha256` and `split_sha256` are blank when the compact archived evidence did not record them.
- `validation_only_selection=true` means that test accuracy was not used to choose the optimizer profile, threshold, or best checkpoint. For fixed-profile runs, it means the test split was evaluated after validation-based checkpoint selection.
- `runner_sha256` identifies the exact benchmark runner used by the source artifact. Several run groups predate the current repository source, so this hash is more precise than assuming that the current file generated every reported value.

## Limitations

The 14 entries in `unsupported_reported_rows.csv` occur in the manuscript table but are not backed by a matching complete three-seed result group in the audited workspace. They should not be presented as artifact-verified until their original selected-run outputs are recovered or the experiments are rerun.

The eight supported MLP-Mixer rows match the August verified reduced-Mixer snapshot. They do not match the later no-bias baseline rerun, so the provenance table deliberately identifies the earlier runner hash rather than combining results from the two implementations. The four global multi-head MLP-Mixer rows appear in the manuscript table, but no corresponding local per-seed or aggregate result artifact was found.

The source manifests contain remote absolute paths and account names. Those manifests were read to verify the experiments but were not copied here. The exported CSV contains only scientific provenance fields and cryptographic hashes; it contains no host names, account names, credentials, or absolute paths.
