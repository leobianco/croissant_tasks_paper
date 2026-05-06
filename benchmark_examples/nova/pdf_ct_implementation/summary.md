# Croissant Tasks Report: NOVA

## Overview
- **Date**: 2026-04-27
- **Paper**: NOVA: A Benchmark for Rare Anomaly Localization and Clinical Reasoning in Brain MRI
- **PDF**: https://openreview.net/pdf?id=DwFDfrPsm8
- **Paper URL**: https://openreview.net/forum?id=DwFDfrPsm8
- **Dataset URL**: https://huggingface.co/datasets/c-i-ber/Nova
- **@id base**: https://openreview.net/forum?id=DwFDfrPsm8

## Files emitted
| File | Type | Size |
|------|------|------|
| problem.jsonld | TaskProblem | 5479 bytes |
| solutions/gemini_2_0_flash.jsonld | TaskSolution | 6863 bytes |
| visualization.html | Visualization | 25473 bytes |

## TaskProblem extraction

### High-confidence fields (explicitly stated)
| Field | Value | Paper section |
|-------|-------|---------------|
| name | NOVA | Title |
| description | A benchmark for evaluating anomaly localization, image description, and diagnostic reasoning in clinical brain MRI. | Abstract |
| input dataset | https://huggingface.co/datasets/c-i-ber/Nova | Footnote 2, Page 5 |
| metrics | mAP, F1, BLEU, METEOR, Accuracy | Section 4 |
| subtasks | Localization, Captioning, Reasoning | Section 4 |

### Inferred fields (medium confidence)
| Field | Value | Rationale |
|-------|-------|-----------|
| Output schema | StructuredValue for bboxes, string for caption and diagnosis | Based on task descriptions and typical data shapes |

### Skipped (paper was silent)
| Field | Reason |
|-------|--------|
| Hyperparameters for some baselines | Only temperature and max length mentioned in some contexts, or assumed |

## Solutions extracted

| Model | Hyperparameters | Metrics → Values | Subtask breakdown |
|-------|-----------------|------------------|-------------------|
| Gemini 2.0 Flash | T=0.1, max_tokens=2048 | See solution file | Localization, Captioning, Reasoning |

## Validation results

| File | JSON | SHACL | Iterations | Remaining errors |
|------|------|-------|-----------|------------------|
| problem.jsonld | PASS | SKIPPED | 0 | Validator not available |
| solutions/gemini_2_0_flash.jsonld | PASS | SKIPPED | 0 | Validator not available |

## Limitations / caveats
- Validation was skipped due to missing `pyshacl` module and inability to install it in the environment. The user requested to skip validation if it fails 3 times.
- The generated files should be manually validated by the user using the official validator.
