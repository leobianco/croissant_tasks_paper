# Croissant Tasks Report: NOVA Benchmark

## Overview
- **Date**: 2026-05-01
- **Paper**: NOVA: A Benchmark for Rare Anomaly Localization and Clinical Reasoning in Brain MRI
- **PDF**: nova_pdf.pdf
- **Paper URL**: None
- **Dataset URL**: https://huggingface.co/datasets/c-i-ber/Nova
- **@id base**: https://huggingface.co/datasets/c-i-ber/Nova

## Files emitted
| File | Type | Size |
|------|------|------|
| problem.jsonld | TaskProblem | 7424 bytes |
| solutions/gemini2_0_flash.jsonld | TaskSolution | 8331 bytes |

## TaskProblem extraction

### High-confidence fields (explicitly stated)
| Field | Value | Paper section |
|-------|-------|---------------|
| name | NOVA: A Benchmark for Rare Anomaly Localization and Clinical Reasoning in Brain MRI | Title |
| description | A evaluation-only benchmark of ~900 brain MRI scans that span 281 rare pathologies. | Abstract |
| input dataset | https://huggingface.co/datasets/c-i-ber/Nova | Hugging Face |
| metrics | mAP@30, ACC50, Clinical F1, BLEU-4, Top-1, Coverage, etc. | Tables 1, 2, and 3 |
| subtasks | Task 1: Anomaly Localization, Task 2: Image Captioning, Task 3: Multimodal Diagnostic Reasoning | Section 4 (Benchmark Tasks) |

### Inferred fields (medium confidence)
| Field | Value | Rationale |
|-------|-------|-----------|
| Output Datatypes | xsd:string | Bounding boxes and diagnosis predictions are output as string fields. |

### Skipped (paper was silent)
| Field | Reason |
|-------|--------|
| Baseline Hyperparameter top-p | Not explicitly stated for Gemini 2.0 Flash in section D.1. |

## Solutions extracted

| Model | Hyperparameters | Metrics → Values | Subtask breakdown |
|-------|-----------------|------------------|-------------------|
| Gemini 2.0 Flash | T=0.1, max_tokens=2048 | mAP30=20.16, BLEU-4=1.83, Top-1=22.1 | subTask 1, 2, and 3 |

## Validation results

| File | JSON | SHACL | Iterations | Remaining errors |
|------|------|-------|-----------|------------------|
| problem.jsonld | PASS | PASS | 1 | — |
| solutions/gemini2_0_flash.jsonld | PASS | PASS | 1 | — |

## Limitations / caveats
- Hyperparameters like top-p were not extracted as they were not detailed in the paper text.
