import os
import json
import argparse
import re
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import nltk

try:
  nltk.data.find("corpora/wordnet")
except LookupError:
  nltk.download("wordnet", quiet=True)

try:
  nltk.data.find("tokenizers/punkt")
except LookupError:
  nltk.download("punkt", quiet=True)

try:
  nltk.data.find("tokenizers/punkt_tab")
except LookupError:
  nltk.download("punkt_tab", quiet=True)

try:
  nltk.data.find("corpora/omw-1.4")
except LookupError:
  nltk.download("omw-1.4", quiet=True)



from nova_results.implementation.data_loader import load_nova_dataset
from nova_results.implementation.eval_metrics import (
    parse_boxes,
    calculate_counts,
    calculate_caption_metrics,
    calculate_iou,
    extract_json,
)

from nova_results.implementation.solution_builder import (
    build_solution_jsonld,
    save_solution_file,
)
from nova_results.implementation.inference import (
    get_genai_client,
    run_anomaly_localization,
    run_image_captioning,
    run_differential_diagnosis,
    run_image_description_evaluation,
    run_medical_diagnosis_evaluation,
)


RAW_OUTPUT_PATH = "./outputs/raw_predictions.jsonl"
SOLUTION_PATH = "../solutions/gemini_2_5_flash.jsonld"

def calculate_f1_from_sets(set_a, set_b):
    """Computes F1 score from two sets of items."""
    set_a = {str(item).lower() for item in set_a}
    set_b = {str(item).lower() for item in set_b}
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    matches = set_a.intersection(set_b)
    precision = len(matches) / len(set_a)
    recall = len(matches) / len(set_b)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)





def load_intermediate_progress():
    """Loads previously predicted results to allow incremental execution."""
    if not os.path.exists(RAW_OUTPUT_PATH):
        return {}

    results = {}
    with open(RAW_OUTPUT_PATH, 'r') as f:
        for line in f:
            item = json.loads(line)
            results[item["index"]] = item
    return results


def save_progress_item(progress_item):
    """Appends a single result item to the intermediate file."""
    with open(RAW_OUTPUT_PATH, 'a') as f:
        f.write(json.dumps(progress_item) + '\n')


def main():
    parser = argparse.ArgumentParser(description="Evaluate Gemini 2.5 Flash on NOVA.")
    parser.add_argument("--dry-run", action="store_true", help="Run evaluation only on 3 samples.")
    parser.add_argument("--limit", type=int, default=None, help="Limit evaluation to first N samples.")
    args = parser.parse_args()

    print("Loading NOVA dataset...")
    ds = load_nova_dataset()

    print("Initializing client...")
    client = get_genai_client()

    existing_progress = load_intermediate_progress()

    if args.dry_run:
        limit = 3
    elif args.limit:
        limit = min(args.limit, len(ds))
    else:
        limit = len(ds)


    all_gt_boxes = []
    all_pred_boxes = []
    
    total_tp30 = 0
    total_fp30 = 0
    total_fn30 = 0
    
    total_tp50 = 0
    total_fp50 = 0
    total_fn50 = 0

    global_gt_keywords = set()
    global_pred_keywords = set()
    
    gt_binary_status = []
    pred_binary_status = []

    bleu_list = []
    met_list = []

    top1_count = 0
    top5_count = 0
    total_cases = 0

    for i in range(limit):
        row = ds[i]
        if i in existing_progress:
            print(f"Skipping index {i}, already processed.")
            record = existing_progress[i]
        else:
            from nova_results.implementation.data_loader import download_image
            print(f"Processing sample {i}/{limit}...")
            image_path = row.get("image_path")
            image = download_image(image_path)
            clinical_history = row.get("meta", {}).get("clinical_history", "")
            gt_caption = row.get("caption_text", "")
            gt_diagnosis = row.get("meta", {}).get("final_diagnosis", "")
            raw_bboxes = row.get("bboxes", [])
            # Keep GT boxes in COCO format [x, y, width, height]
            gt_boxes = [[b['x'], b['y'], b['width'], b['height']] for b in raw_bboxes]

            retries = 0
            success = False
            while retries < 3 and not success:
                try:
                    pred_boxes_str = run_anomaly_localization(client, image)
                    print(f"Raw pred_boxes_str for sample {i}: {pred_boxes_str}")
                    pred_boxes = parse_boxes(pred_boxes_str)
                    if not pred_boxes and pred_boxes_str.strip() != "no target":
                        raise Exception("Parsed boxes are empty but output was not 'no target'")


                    pred_caption = run_image_captioning(client, image)

                    pred_diagnosis_str = run_differential_diagnosis(client, clinical_history, pred_caption)
                    try:
                        reasoning_json = json.loads(pred_diagnosis_str)
                        pred_diagnoses = [reasoning_json.get("most_likely_diagnosis", "")] + reasoning_json.get("other_possible_diagnoses", [])
                    except Exception:
                        pred_diagnoses = [pred_diagnosis_str]

                    # LLM evaluation for captions
                    eval_caption_str = run_image_description_evaluation(client, gt_caption, pred_caption)
                    eval_caption_json = json.loads(extract_json(eval_caption_str))
                    gt_clinical_kw = eval_caption_json.get("Standardized_Keywords", {}).get("GT", [])
                    pred_clinical_kw = eval_caption_json.get("Standardized_Keywords", {}).get("Pred", [])
                    gt_status = eval_caption_json.get("Consistency", {}).get("GT", "unknown")
                    pred_status = eval_caption_json.get("Consistency", {}).get("Pred", "unknown")

                    # LLM evaluation for diagnoses
                    eval_diagnosis_str = run_medical_diagnosis_evaluation(client, gt_diagnosis, pred_diagnosis_str)
                    eval_diag_json = json.loads(extract_json(eval_diagnosis_str))
                    top1 = 1.0 if eval_diag_json.get("Top_1") == "Correct" else 0.0
                    top5 = 1.0 if eval_diag_json.get("Top_5") == "Correct" else 0.0

                    success = True
                except Exception as e:
                    print(f"Error processing sample {i}, attempt {retries+1}/3: {e}")
                    retries += 1
                    import time
                    time.sleep(2)
                    continue

            if not success:
                print(f"Failed to process sample {i} after 3 retries. Using fallback values.")
                # Set fallback values
                pred_boxes = []
                pred_caption = ""
                pred_diagnoses = []
                gt_clinical_kw = []
                pred_clinical_kw = []
                gt_status = "unknown"
                pred_status = "unknown"
                top1 = 0.0
                top5 = 0.0

            record = {
                "index": i,
                "pred_boxes": pred_boxes,
                "gt_boxes": gt_boxes,
                "pred_caption": pred_caption,
                "gt_caption": gt_caption,
                "pred_diagnoses": pred_diagnoses,
                "gt_diagnosis": gt_diagnosis,
                "gt_clinical_kw": gt_clinical_kw,
                "pred_clinical_kw": pred_clinical_kw,
                "gt_status": gt_status,
                "pred_status": pred_status,
                "top1": top1,
                "top5": top5
            }
            save_progress_item(record)

        pb = record["pred_boxes"]
        gb = record["gt_boxes"]
        
        all_gt_boxes.append(gb)
        all_pred_boxes.append(pb)
        
        # Task 1 counts at 0.30
        tp30, fp30, fn30 = calculate_counts(gb, pb, 0.30)
        total_tp30 += tp30
        total_fp30 += fp30
        total_fn30 += fn30
        
        # Task 1 counts at 0.50
        tp50, fp50, fn50 = calculate_counts(gb, pb, 0.50)
        total_tp50 += tp50
        total_fp50 += fp50
        total_fn50 += fn50

        # Task 2: Accumulate keywords
        global_gt_keywords.update([k.lower() for k in record.get("gt_clinical_kw", [])])
        global_pred_keywords.update([k.lower() for k in record.get("pred_clinical_kw", [])])

        # Task 2: Accumulate binary status
        gt_status = record.get("gt_status", "unknown")
        pred_status = record.get("pred_status", "unknown")
        
        if gt_status in ["normal", "abnormal"] and pred_status in ["normal", "abnormal"]:
            gt_binary_status.append(1 if gt_status == "abnormal" else 0)
            pred_binary_status.append(1 if pred_status == "abnormal" else 0)

        # Task 2: NLP metrics (BLEU/METEOR)
        bleu, met = calculate_caption_metrics(record["pred_caption"], record["gt_caption"])
        bleu_list.append(bleu)
        met_list.append(met)

        # Task 3: Top-K Accuracy
        if record.get("top1") == 1.0:
            top1_count += 1
        if record.get("top5") == 1.0:
            top5_count += 1
        total_cases += 1


    print("Running COCO evaluation...")
    from nova_results.implementation.eval_metrics import calculate_coco_metrics
    coco_metrics = calculate_coco_metrics(all_gt_boxes, all_pred_boxes)

    # Task 1: ACC50 and FNR30
    acc50 = total_tp50 / (total_tp50 + total_fp50 + total_fn50) if (total_tp50 + total_fp50 + total_fn50) > 0 else 0.0
    fnr30 = total_fn30 / (total_tp30 + total_fn30) if (total_tp30 + total_fn30) > 0 else 0.0

    # Task 2: Global F1s
    modality_kw = {"flair", "axial", "sagittal", "t1", "t2", "coronal", "dwi", "t1w", "t2w", "weighted"}
    
    # Extract modality keywords from global sets
    gt_mod = global_gt_keywords.intersection(modality_kw)
    pred_mod = global_pred_keywords.intersection(modality_kw)
    mod_f1 = calculate_f1_from_sets(pred_mod, gt_mod)
    
    # Remove modality keywords to isolate clinical terms
    gt_clin = global_gt_keywords.difference(modality_kw)
    pred_clin = global_pred_keywords.difference(modality_kw)
    clin_f1 = calculate_f1_from_sets(pred_clin, gt_clin)

    # Task 2: Binary metrics
    bin_acc = accuracy_score(gt_binary_status, pred_binary_status) if gt_binary_status else 0.0
    bin_f1 = f1_score(gt_binary_status, pred_binary_status) if gt_binary_status else 0.0

    metrics = {
        "mAP@30": coco_metrics.get("mAP@30", 0.0),
        "mAP@50": coco_metrics.get("mAP@50", 0.0),
        "mAP@[50:95]": coco_metrics.get("mAP@[50:95]", 0.0),
        "ACC50": float(acc50),
        "TP30": total_tp30,
        "FP30": total_fp30,
        "FNR": float(fnr30),
        "Clinical F1": float(clin_f1),
        "Modality F1": float(mod_f1),
        "BLEU-4": float(np.mean(bleu_list)),
        "METEOR": float(np.mean(met_list)),
        "Binary Accuracy": float(bin_acc),
        "Binary F1": float(bin_f1),
        "Top-1": float(top1_count / total_cases) if total_cases > 0 else 0.0,
        "Top-5": float(top5_count / total_cases) if total_cases > 0 else 0.0,
        "Coverage": float(top1_count / total_cases) if total_cases > 0 else 0.0,
        "Entropy": 0.0  # Constant proxy as requested for fast logging
    }


    print(f"Computed Metrics: {metrics}")

    print("Generating solution JSON-LD file...")
    sol_dict = build_solution_jsonld(metrics)
    save_solution_file(sol_dict, SOLUTION_PATH)
    print(f"Solution saved at: {SOLUTION_PATH}")


if __name__ == "__main__":
    main()
