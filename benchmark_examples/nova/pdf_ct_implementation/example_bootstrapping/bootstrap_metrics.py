import json
import os
from tqdm import tqdm
import evaluate
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from sklearn.metrics import accuracy_score, f1_score
from nova_results.implementation.run_evaluation import calculate_f1_from_sets
from nova_results.implementation.eval_metrics import calculate_coco_metrics, calculate_counts


RAW_OUTPUT_PATH = "./outputs/raw_predictions.jsonl"
BOOTSTRAP_PATH = "./outputs/bootstrap_iterations.jsonl"

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
                continue

    print(f"Total records: {len(records)}")

    bleu = evaluate.load("bleu")
    meteor = evaluate.load("meteor")

    n_iterations = 1000
    n_samples = len(records)

    # Load existing bootstrap results if available
    existing_results = []
    if os.path.exists(BOOTSTRAP_PATH):
        print(f"Loading existing bootstrap results from {BOOTSTRAP_PATH}...")
        with open(BOOTSTRAP_PATH, "r") as f:
            for line in f:
                existing_results.append(json.loads(line))
        print(f"Loaded {len(existing_results)} iterations.")

    new_results = []
    
    print("Starting bootstrap...")
    for i in tqdm(range(n_iterations)):
        if i < len(existing_results):
            res = existing_results[i]
            sampled_indices = res.get("sampled_indices")
            metrics = res.get("metrics", {})
        else:
            # Sample indices with replacement (only if not loaded)
            sampled_indices = np.random.choice(n_samples, size=n_samples, replace=True).tolist()
            metrics = {}

        sampled_records = [records[idx] for idx in sampled_indices]
        
        # Task 1: Localization
        if "mAP@30" not in metrics or "ACC50" not in metrics:
            loc_records = [r for r in sampled_records if len(r.get("pred_boxes", [])) > 0]
            if loc_records:
                all_gt_boxes = [r.get("gt_boxes", []) for r in loc_records]
                all_pred_boxes = [r.get("pred_boxes", []) for r in loc_records]
                coco_metrics = calculate_coco_metrics(all_gt_boxes, all_pred_boxes)
                
                # ACC50 and FNR
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
                
            metrics.update(coco_metrics)
            metrics["ACC50"] = acc50
            metrics["FNR"] = fnr

        # Task 2: Captioning
        if "BLEU-4" not in metrics:
            cap_records = [r for r in sampled_records if r.get("pred_caption", "") != ""]
            if cap_records:
                predictions = [r.get("pred_caption", "") for r in cap_records]
                references = [[r.get("gt_caption", "")] for r in cap_records]
                bleu_result = bleu.compute(predictions=predictions, references=references)
                meteor_result = meteor.compute(predictions=predictions, references=references)
            else:
                bleu_result = {"bleu": 0.0}
                meteor_result = {"meteor": 0.0}
                
            metrics["BLEU-4"] = bleu_result["bleu"]
            metrics["METEOR"] = meteor_result["meteor"]

        # Task 2: Keyword F1 & Binary Metrics
        if "Clinical_F1" not in metrics:
            f1_records = [r for r in sampled_records if r.get("gt_status") != "unknown"]
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
                metrics["Modality_F1"] = calculate_f1_from_sets(pred_mod, gt_mod)
                
                gt_clin = global_gt_keywords.difference(modality_kw)
                pred_clin = global_pred_keywords.difference(modality_kw)
                metrics["Clinical_F1"] = calculate_f1_from_sets(pred_clin, gt_clin)
                
                metrics["Binary_Accuracy"] = accuracy_score(gt_statuses, pred_statuses) if gt_statuses else 0.0
                pos_label = 'abnormal' if 'abnormal' in gt_statuses else None
                metrics["Binary_F1"] = f1_score(gt_statuses, pred_statuses, pos_label=pos_label, average='binary') if gt_statuses and pos_label else 0.0
            else:
                metrics["Modality_F1"] = 0.0
                metrics["Clinical_F1"] = 0.0
                metrics["Binary_Accuracy"] = 0.0
                metrics["Binary_F1"] = 0.0

        # Task 3: Reasoning
        if "Top-1" not in metrics:
            reas_records = [r for r in sampled_records if len(r.get("pred_diagnoses", [])) > 0]
            if reas_records:
                metrics["Top-1"] = sum([r.get("top1", 0.0) for r in reas_records]) / len(reas_records)
                metrics["Top-5"] = sum([r.get("top5", 0.0) for r in reas_records]) / len(reas_records)
            else:
                metrics["Top-1"] = 0.0
                metrics["Top-5"] = 0.0

        new_results.append({

            "iteration": i,
            "sampled_indices": sampled_indices,
            "metrics": metrics
        })

    print(f"Saving bootstrap results to {BOOTSTRAP_PATH}...")
    with open(BOOTSTRAP_PATH, "w") as f:
        for res in new_results:
            f.write(json.dumps(res) + "\n")

    # Calculate and print CI
    metrics_history = {}
    for res in new_results:
        for k, v in res["metrics"].items():
            if k not in metrics_history:
                metrics_history[k] = []
            metrics_history[k].append(v)

    print("\n--- Bootstrap Results (95% Confidence Intervals) ---")
    for metric, values in metrics_history.items():
        mean_val = np.mean(values)
        lower_bound = np.percentile(values, 2.5)
        upper_bound = np.percentile(values, 97.5)
        print(f"{metric}: Mean={mean_val:.4f}, 95% CI=[{lower_bound:.4f}, {upper_bound:.4f}]")

if __name__ == "__main__":
    main()
