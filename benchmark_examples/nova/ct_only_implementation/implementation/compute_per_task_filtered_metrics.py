import json
import os
from sklearn.metrics import accuracy_score, f1_score
import evaluate
import numpy as np

from nova_results.implementation.eval_metrics import calculate_coco_metrics, calculate_counts
from nova_results.implementation.run_evaluation import calculate_f1_from_sets


RAW_OUTPUT_PATH = "./outputs/raw_predictions.jsonl"

def main():
    if not os.path.exists(RAW_OUTPUT_PATH):
        print(f"File not found: {RAW_OUTPUT_PATH}")
        return

    print(f"Loading predictions from {RAW_OUTPUT_PATH}...")
    records = []
    with open(RAW_OUTPUT_PATH, "r") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print("Skipping incomplete or malformed JSON line.")
                continue


    print(f"Total records: {len(records)}")

    # Task 1: Localization
    # Filter records with non-empty pred_boxes
    loc_records = [r for r in records if len(r.get("pred_boxes", [])) > 0]
    print(f"Records for Localization: {len(loc_records)}")
    
    if loc_records:
        all_gt_boxes = [r.get("gt_boxes", []) for r in loc_records]
        all_pred_boxes = [r.get("pred_boxes", []) for r in loc_records]
        coco_metrics = calculate_coco_metrics(all_gt_boxes, all_pred_boxes)
        
        tp50, fp50, fn50 = 0, 0, 0
        for gt_b, pred_b in zip(all_gt_boxes, all_pred_boxes):
            tp, fp, fn = calculate_counts(gt_b, pred_b, iou_threshold=0.5)
            tp50 += tp
            fp50 += fp
            fn50 += fn
            
        acc50 = tp50 / (tp50 + fp50 + fn50) if (tp50 + fp50 + fn50) > 0 else 0.0
        fnr = fn50 / (tp50 + fn50) if (tp50 + fn50) > 0 else 0.0
    else:
        coco_metrics = {"mAP@30": 0.0, "mAP@50": 0.0, "mAP@[50:95]": 0.0}
        acc50, fnr = 0.0, 1.0


    # Task 2: Captioning
    bleu = evaluate.load("bleu")
    meteor = evaluate.load("meteor")

    # Filter records for BLEU/METEOR (non-empty pred_caption)
    cap_records = [r for r in records if r.get("pred_caption", "") != ""]
    print(f"Records for Caption BLEU/METEOR: {len(cap_records)}")

    if cap_records:
        predictions = [r.get("pred_caption", "") for r in cap_records]
        references = [[r.get("gt_caption", "")] for r in cap_records]
        bleu_result = bleu.compute(predictions=predictions, references=references)
        meteor_result = meteor.compute(predictions=predictions, references=references)
    else:
        bleu_result = {"bleu": 0.0}
        meteor_result = {"meteor": 0.0}

    # Filter records for Global F1 and Binary Metrics (status not unknown)
    f1_records = [r for r in records if r.get("gt_status") != "unknown"]
    print(f"Records for Keyword F1 & Binary Metrics: {len(f1_records)}")

    if f1_records:
        global_gt_keywords = set()
        global_pred_keywords = set()
        gt_statuses = []
        pred_statuses = []

        for r in f1_records:
            global_gt_keywords.update([k.lower() for k in r.get("gt_clinical_kw", [])])
            global_pred_keywords.update([k.lower() for k in r.get("pred_clinical_kw", [])])
            
            if r.get("pred_status") != "unknown":
                gt_statuses.append(r.get("gt_status"))
                pred_statuses.append(r.get("pred_status"))

        modality_kw = {"flair", "axial", "sagittal", "t1", "t2", "coronal", "dwi", "t1w", "t2w", "weighted"}
        
        gt_mod = global_gt_keywords.intersection(modality_kw)
        pred_mod = global_pred_keywords.intersection(modality_kw)
        mod_f1 = calculate_f1_from_sets(pred_mod, gt_mod)
        
        gt_clin = global_gt_keywords.difference(modality_kw)
        pred_clin = global_pred_keywords.difference(modality_kw)
        clin_f1 = calculate_f1_from_sets(pred_clin, gt_clin)

        bin_acc = accuracy_score(gt_statuses, pred_statuses) if gt_statuses else 0.0
        pos_label = 'abnormal' if 'abnormal' in gt_statuses else None
        bin_f1 = f1_score(gt_statuses, pred_statuses, pos_label=pos_label, average='binary') if gt_statuses and pos_label else 0.0
    else:
        clin_f1, mod_f1, bin_acc, bin_f1 = 0.0, 0.0, 0.0, 0.0

    # Task 3: Reasoning
    # Filter records with non-empty pred_diagnoses
    reas_records = [r for r in records if len(r.get("pred_diagnoses", [])) > 0]
    print(f"Records for Reasoning: {len(reas_records)}")

    if reas_records:
        top1_acc = sum([r.get("top1", 0.0) for r in reas_records]) / len(reas_records)
        top5_acc = sum([r.get("top5", 0.0) for r in reas_records]) / len(reas_records)
    else:
        top1_acc, top5_acc = 0.0, 0.0

    print("\n--- Per-Task Filtered Metrics ---")
    print(f"mAP@30: {coco_metrics['mAP@30']}")
    print(f"mAP@50: {coco_metrics['mAP@50']}")
    print(f"mAP@[50:95]: {coco_metrics['mAP@[50:95]']}")
    print(f"ACC50: {acc50}")
    print(f"FNR: {fnr}")
    print(f"Clinical F1: {clin_f1}")
    print(f"Modality F1: {mod_f1}")
    print(f"BLEU-4: {bleu_result['bleu']}")
    print(f"METEOR: {meteor_result['meteor']}")
    print(f"Binary Accuracy: {bin_acc}")
    print(f"Binary F1: {bin_f1}")
    print(f"Top-1 Accuracy: {top1_acc}")
    print(f"Top-5 Accuracy: {top5_acc}")


if __name__ == "__main__":
    main()
