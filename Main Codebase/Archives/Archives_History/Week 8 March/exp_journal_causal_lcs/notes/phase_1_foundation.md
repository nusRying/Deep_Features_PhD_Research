# Phase 1 Foundation Notes

This experiment keeps ExSTraCS modifications local so the journal branch remains independent.

Implemented foundation hooks:

- `DefaultFitnessPolicy`
- `DefaultParentSelectionPolicy`
- `DefaultSubsumptionPolicy`
- `initial_population` support in the local `ExSTraCS` constructor

Design intent:

- future causal logic should be injected through policy objects instead of rewriting the local core loop again
- later experiments can be cloned from this folder if they need a new independent branch