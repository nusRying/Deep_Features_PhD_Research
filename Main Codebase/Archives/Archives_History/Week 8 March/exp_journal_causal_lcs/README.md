# Experiment: Journal Causal-LCS

This experiment is an isolated journal-development branch for the Causal-LCS pathway.

It is intentionally independent from the rest of the repository:

- it has its own local `external/scikit-ExSTraCS-master` snapshot
- all ExSTraCS hook work for this experiment stays inside this folder
- future Causal-LCS code changes should be made here, not in the shared root copy

## Phase 1 Status

Phase 1 implements the local extension foundation only.

Current local changes:

- pluggable fitness policy
- pluggable parent-selection policy
- pluggable subsumption policy
- optional local seeded initial population support

These hooks preserve the default ExSTraCS behavior unless a custom policy object is passed to `ExSTraCS(...)`.

## Local files of interest

- `external/scikit-ExSTraCS-master/skExSTraCS/ExtensionHooks.py`
- `external/scikit-ExSTraCS-master/skExSTraCS/ExSTraCS.py`
- `external/scikit-ExSTraCS-master/skExSTraCS/Classifier.py`
- `external/scikit-ExSTraCS-master/skExSTraCS/ClassifierSet.py`

## Next Phase

Phase 2 should add the Causal-LCS-specific modules in this experiment only:

- causal metadata generation
- causal fitness calculator
- causal subsumption validator
- experiment-specific configuration and notes

Phase 2 scaffolding is now present in:

- `causal_lcs/config.py`
- `causal_lcs/io.py`
- `causal_lcs/metadata.py`
- `causal_lcs/fitness.py`
- `causal_lcs/subsumption.py`
- `main.py`

These files are dataset-aware for the current repo layout and have not been executed.

## Prepared Execution Path

The local entrypoint now supports two modes when you decide to run it yourself:

- `--prepare-only`: build metadata and split artifacts without training
- `--train`: start a single local Causal-LCS training run

Train mode now writes:

- `models/causal_lcs_model.pkl`
- `results/train_run_summary.json`
- `results/tracking/iterationData.csv`

No commands have been executed by me.