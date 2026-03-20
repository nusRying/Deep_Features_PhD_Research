# Phase 1: LCS Baseline (Single Expert)

## Overview

This experiment establishes the initial baseline performance of a single ExSTraCS expert on EfficientNet deep features.

## Key Goals

- Validate the connectivity between deep feature extraction and the LCS engine.
- Establish the first "Internal BA" record for comparison with SOTA methods.

## Results Summary

- **Balanced Accuracy**: ~68-72% (Early Milestone)
- **Contribution**: Proved that LCS can learn directly from high-dimensional PCA-reduced CNN features.

## Reproduction

```powershell
python main.py
```
