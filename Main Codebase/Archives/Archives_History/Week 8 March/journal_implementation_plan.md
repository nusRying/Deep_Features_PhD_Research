# Journal Publication Implementation Plan: ExSTraCS Paradigm Shifts

This document converts the earlier concept sketch into an implementation plan that is actually compatible with the current codebase.

The main change in strategy is simple: do not start by rewriting ExSTraCS internals for three architectures at once. First create a small extension layer in the canonical ExSTraCS copy, then pursue one journal-ready architecture deeply.

---

## Recommended Journal Path

### Primary recommendation: Causal-LCS (C-LCS)

This is the strongest first paper candidate because it offers the cleanest scientific contribution relative to the current project state:

- It builds directly on the existing knowledge-discovery direction from Phase 7.
- It can be framed as a principled upgrade from associative rule learning to causally adjusted rule valuation.
- It requires fewer moving parts than Graph-LCS.
- It is easier to defend methodologically than an LLM-seeding paper, where reviewers can question prompt sensitivity and reproducibility.

### Secondary options

- Neuro-Symbolic LLM-LCS (NS-LCS): worthwhile as a follow-up paper or ablation branch.
- Graph-LCS (G-LCS): highest novelty, but also highest engineering and runtime risk. This should be treated as a later research program, not the first implementation target.

---

## Core Problem With the Original Plan

The original draft assumed that ExSTraCS can accept new fitness logic, new parent selection logic, seeded populations, and new subsumption criteria with small direct edits. That is not how the current codebase is structured.

In the canonical copy at `external/scikit-ExSTraCS-master/skExSTraCS/`:

- fitness is hardcoded inside `Classifier.updateFitness()`
- prediction voting is fitness-weighted inside `Prediction`
- parent selection is hardwired inside `ClassifierSet.runGA()`
- subsumption uses syntax-only generality checks in `Classifier.isMoreGeneral()`
- population initialization does not expose a clean seeded-population interface

This means all three proposed architectures depend on the same missing prerequisite: a minimal extension layer.

---

## Phase 0: Architecture Selection Gate

Before any code changes, formally select one architecture using explicit criteria.

### Selection criteria

Score each architecture on:

1. Novelty for a journal submission
2. Feasibility within the current ExSTraCS code structure
3. Runtime cost
4. Reproducibility
5. Strength of ablation story
6. Fit with existing Phase 7 knowledge-discovery outputs

### Decision

Proceed with Causal-LCS first unless new evidence shows that the causal assumptions are too weak to support a defensible experiment.

---

## Phase 1: Build the ExSTraCS Extension Layer

This is the real prerequisite for every architecture.

### Objective

Introduce a minimal hook system into the canonical ExSTraCS copy so that future architectures do not require repeated invasive forks of core logic.

### Target codebase

Use the canonical path:

`external/scikit-ExSTraCS-master/skExSTraCS/`

Do not begin from one of the older phase-local copies unless a later experiment requires an isolated frozen snapshot.

### Required hooks

1. Fitness hook
   - Replace direct fitness assignment with a model-level calculator.
   - Default behavior must remain `accuracy^nu`.

2. Parent selection hook
   - Extract parent selection from `ClassifierSet.runGA()` into a selector interface.
   - Default behavior must preserve current roulette and tournament behavior.

3. Subsumption validation hook
   - Keep current syntactic generality as the default.
   - Add a validator layer so causal or graph-aware checks can be inserted without rewriting `Classifier`.

4. Initial population hook
   - Allow optional seeded rules before training begins.
   - Default behavior must remain an empty starting population.

5. Tracking hook
   - Extend tracking so new metrics can be written alongside the existing iteration record exports.

### Deliverables

- one extension module for default and custom policies
- regression test notes showing baseline behavior is unchanged when defaults are used
- a short developer README describing how new architectures plug into the hooks

### Success condition

Running the existing baseline behavior through the new hooks produces materially unchanged outputs.

---

## Phase 2: Create a Standalone Journal Experiment Track

Follow the repository's established pattern of isolated experimental phases instead of modifying the existing production phases directly.

### Recommended structure

Create a dedicated experiment directory under Week 8 March, for example:

`Week 8 March/exp_journal_causal_lcs/`

This directory should contain:

- `main.py`
- `README.md`
- `requirements.txt` if needed
- `configs/`
- `results/`
- `notes/`
- `external/` only if the canonical ExSTraCS snapshot must be frozen for paper reproducibility

### Rationale

This keeps the journal pathway reproducible and consistent with the repo's phase-based workflow while avoiding accidental disruption of existing pipelines.

---

## Phase 3: Implement Causal-LCS

## Objective

Replace purely associative rule valuation with causally adjusted valuation while preserving the ExSTraCS rule-evolution machinery as much as possible.

### Causal-LCS should be implemented in three layers

### 3.1 Causal data layer

This should happen before ExSTraCS training starts.

Required outputs:

- a declared causal graph or a justified approximation of one
- confounder definitions
- treatment feature groups or rule-level treatment interpretation strategy
- propensity scores or inverse-probability weights
- a serialized metadata file consumed by the training pipeline

Important constraint:

Do not hide all causal assumptions inside `main.py`. They need to be explicit, versioned, and saved as experiment artifacts.

### 3.2 Causal fitness layer

Replace the direct `accuracy^nu` update with a mixed objective:

$$
Fitness(R) = \alpha \cdot CausalScore(R) + (1 - \alpha) \cdot Accuracy(R)^{\nu}
$$

Where `CausalScore(R)` should be clearly defined in the experiment README. If a full rule-level ATE estimate is too unstable, start with a simpler causally weighted correctness score before attempting a full do-calculus claim.

Recommended implementation note:

- begin with a cached causal score per rule update window rather than recomputing a full estimate on every micro-update

### 3.3 Causal subsumption layer

The original idea is directionally correct but needs to be stated more carefully.

New rule:

- a rule may only subsume another rule if standard ExSTraCS subsumption conditions are met
- and both rules are compatible under the declared causal parent structure

This avoids merging rules that are syntactically similar but causally inconsistent.

### Minimal viable Causal-LCS experiment

The first journal prototype should include:

- default hook layer
- causal metadata generation
- causal fitness calculator
- causal subsumption validator
- comparison against Phase 7 baseline metrics

Do not attempt to redesign prediction voting in the first iteration unless the fitness change clearly proves insufficient.

---

## Phase 4: Experimental Protocol for Causal-LCS

The paper will be much stronger if the experiment protocol is fixed now rather than after implementation.

### Baseline comparisons

At minimum compare against:

1. canonical or current ExSTraCS baseline
2. Phase 7 knowledge-discovery configuration
3. strongest finalized hybrid baseline already reported in this repo

### Evaluation outputs

Track:

- balanced accuracy
- sensitivity
- specificity
- ROC-AUC if available in the current pipeline
- rule population size
- rule stability across seeds
- rule interpretability summaries
- causal consistency diagnostics

### Reproducibility requirements

- fixed seed list
- smoke-test mode for quick verification
- saved causal metadata artifacts
- saved run config
- saved tracking CSV

### Required ablations

At minimum include:

1. baseline fitness only
2. causal fitness only
3. mixed fitness with varying `alpha`
4. causal subsumption on vs off

These ablations are more valuable for the paper than simultaneously implementing a second architecture.

---

## Phase 5: Neuro-Symbolic LLM-LCS as a Follow-Up Branch

This should be treated as a second branch after the extension layer exists.

### Objective

Seed the initial population with literature-inspired rules while keeping the rest of training reproducible and measurable.

### What must change from the original plan

- do not describe this as a simple replacement of random initialization
- require a seeded-population interface in ExSTraCS first
- record rule provenance explicitly, such as `seed_source = llm`
- preserve a strict parser from LLM output into rule objects
- benchmark convergence speed and final quality separately

### Minimum viable NS-LCS experiment

- LLM rule generator or curated static seed file
- parser and schema validator
- seeded population initialization hook
- survival policy for seeded rules during early generations
- comparison against unseeded training under the same seeds

### Paper risk to acknowledge early

If the LLM component is used, the experiment must control for prompt version, model version, and parsing failures. Otherwise the reproducibility story weakens significantly.

---

## Phase 6: Graph-LCS as a Longer-Term Research Track

This remains interesting, but it should not be the first implementation target.

### Objective

Represent the rule population as a graph and use graph structure to influence evolution and compaction.

### Why it is high risk

- current populations are flat lists, not graph-backed structures
- overlap graph construction can become expensive
- graph updates must stay synchronized with covering, mutation, crossover, deletion, and subsumption
- centrality-aware subsumption introduces both correctness and performance risks

### What would make Graph-LCS viable later

- a dedicated population graph abstraction
- lazy adjacency updates
- clear complexity controls
- proof that graph-guided parent selection improves something meaningful beyond novelty

### Recommendation

Keep this in the document as a future direction or secondary proposal, not as part of the first journal build.

---

## Execution Strategy

The implementation order should be:

1. Select one architecture formally
2. Build the ExSTraCS extension layer
3. Verify regression compatibility with current default behavior
4. Create a standalone journal experiment directory
5. Implement Causal-LCS only
6. Run baseline comparisons and ablations
7. Decide whether NS-LCS is needed as a second contribution or supporting experiment

This is the shortest path to a paper-quality result.

---

## Concrete Deliverables

### Engineering deliverables

- minimal ExSTraCS hook layer
- standalone journal experiment folder
- causal metadata generation pipeline
- causal fitness implementation
- causal subsumption implementation
- smoke-test execution path

### Experiment deliverables

- seed-controlled benchmark runs
- ablation tables
- per-run tracking CSVs
- final metrics summary JSON
- rule reports comparable to current knowledge-discovery outputs

### Manuscript deliverables

- method diagram
- causal assumptions table
- ablation results table
- performance comparison table
- example discovered rules with causal interpretation

---

## Final Recommendation

The best path is not to build all three architectures.

The best path is:

- treat Causal-LCS as the primary journal target
- first create a small ExSTraCS extension layer
- run one rigorous standalone experiment branch with ablations
- position NS-LCS and G-LCS as follow-up work unless the causal route fails empirically

That strategy is better aligned with the current repository, more defensible in a paper, and much more likely to finish cleanly.
