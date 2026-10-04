# Release checklist

The code and reproducibility metadata are prepared locally on the
`codex/paper-release` branch. Before publishing a tagged release:

- choose and add a software license; no license has been selected on the
  authors' behalf;
- either rerun the strictly matched `ff-matched-local` control or keep the
  manuscript wording explicit that `ff` versus `ff-matched-ge` compares two
  complete training formulations, including optimizer topology and separately
  selected thresholds;
- confirm that the title and author order in `CITATION.cff` match the submitted
  manuscript;
- update `CITATION.cff` with the article DOI and release date when available;
- compare `paper_results/reported_accuracy.csv` with the final accepted tables;
- recover or rerun the 14 entries listed in
  `paper_results/unsupported_reported_rows.csv`, which are currently
  documented only by their manuscript aggregates;
- run the complete unit-test workflow in a clean Python 3.10 environment;
- tag the exact source revision used for any final reruns and archive the run
  manifests with the paper artifacts.
