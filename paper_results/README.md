# Paper-table result snapshot

`reported_accuracy.csv` contains the aggregate test accuracies supplied for
this release snapshot. Values are percentages reported as the mean and sample
standard deviation across seeds 424, 425, and 426. Reconcile this snapshot
against the final manuscript before tagging a public release.

These rows are a transcription of the supplied paper tables, not fresh outputs produced
when this repository is installed. The executable schedulers write complete
run manifests, per-seed records, profile-selection evidence, and summaries into
their requested run directories. Candidate optimizer profiles are selected by
validation performance; the test split is evaluated only for the selected
configuration.

`per_seed_provenance.csv` contains sanitized per-seed evidence for 50 of the
64 reported aggregate rows. `unsupported_reported_rows.csv` identifies the 14
reported rows for which no matching complete local artifact was found during
the release audit. See `PROVENANCE.md` for the matching rule, source-run groups,
available hashes, and limitations. A manuscript aggregate is not described as
artifact-verified unless its three per-seed values reproduce both the reported
mean and sample standard deviation after two-decimal rounding.

Blank values with status `not_evaluated` are intentional. The full-comparison
FF family is omitted on CIFAR-100 because it requires 100 class-conditioned
evaluations for every example.
