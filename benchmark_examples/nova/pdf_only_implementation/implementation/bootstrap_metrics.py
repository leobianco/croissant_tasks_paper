import json
import os
import random
import evaluate
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

FULL_RESULTS_PATH = "./outputs/full_results.json"
BOOTSTRAP_PATH = "./outputs/bootstrap_results.jsonl"
OUTPUT_DIR = "./outputs"

def calculate_iou(box_a, box_b):
  """Calculates Intersection over Union (IoU) between two boxes."""
  y_a = max(box_a[0], box_b[0])
  x_a = max(box_a[1], box_b[1])
  y_b = min(box_a[2], box_b[2])
  x_b = min(box_a[3], box_b[3])

  inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)

  box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
  box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

  iou = inter_area / float(box_a_area + box_b_area - inter_area)
  return iou

def evaluate_localization_manual(preds, golds, iou_threshold=0.5):
  """Computes Precision, Recall, and F1 for localization at a given IoU threshold."""
  if not golds:
    return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 0}

  tp = 0
  fp = 0
  matched_golds = set()

  for pred in preds:
    pred_box = pred.get("bbox_2d") or pred.get("box_2d")
    if not pred_box:
      continue

    best_iou = 0
    best_gold_idx = -1

    for idx, gold in enumerate(golds):
      if idx in matched_golds:
        continue
      gold_box = [gold["y"], gold["x"], gold["y"] + gold["height"], gold["x"] + gold["width"]]
      iou = calculate_iou(pred_box, gold_box)
      if iou > best_iou:
        best_iou = iou
        best_gold_idx = idx

    if best_iou >= iou_threshold:
      tp += 1
      matched_golds.add(best_gold_idx)
    else:
      fp += 1

  fn = len(golds) - len(matched_golds)

  precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
  recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
  f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

  return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

def compute_f1(gt_set, pred_set):
  """Computes F1 score between two sets of items."""
  intersection = gt_set.intersection(pred_set)
  precision = len(intersection) / len(pred_set) if len(pred_set) > 0 else 0.0
  recall = len(intersection) / len(gt_set) if len(gt_set) > 0 else 0.0
  return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

def compute_macro_f1(y_true, y_pred):
  def f1_for_class(cls):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

  return (f1_for_class(0) + f1_for_class(1)) / 2

def main():
  if not os.path.exists(FULL_RESULTS_PATH):
    print(f"File not found: {FULL_RESULTS_PATH}")
    return

  print(f"Loading results from {FULL_RESULTS_PATH}...")
  with open(FULL_RESULTS_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)
    results = data.get("results", [])

  print(f"Total records: {len(results)}")

  bleu_metric = evaluate.load("bleu")
  meteor_metric = evaluate.load("meteor")

  modality_terms = {"flair", "axial", "sagittal", "t1", "t2", "coronal", "dwi", "t1w", "t2w", "weighted"}

  n_iterations = 1000
  n_samples = len(results)

  existing_results = []
  if os.path.exists(BOOTSTRAP_PATH):
    print(f"Loading existing bootstrap results from {BOOTSTRAP_PATH}...")
    with open(BOOTSTRAP_PATH, "r") as f:
      for line in f:
        existing_results.append(json.loads(line))
    print(f"Loaded {len(existing_results)} iterations.")

  new_results = []

  print("Starting bootstrap...")
  for i in range(n_iterations):
    if i % 10 == 0:
      print(f"Iteration {i}/{n_iterations}")
    if i < len(existing_results):
      res = existing_results[i]
      sampled_indices = res.get("sampled_indices")
      metrics = res.get("metrics", {})
    else:
      sampled_indices = random.choices(range(n_samples), k=n_samples)
      metrics = {}

    sampled_records = [results[idx] for idx in sampled_indices]

    # 1. Localization Metrics (ACC50, FNR, mAP)
    if "ACC50" not in metrics or "mAP@[50:95]" not in metrics:
      tp50, fp50, fn50 = 0, 0, 0
      tp30, fp30, fn30 = 0, 0, 0
      coco_gt = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "anomaly"}]}
      coco_dt = []
      ann_id = 1

      for idx, r in enumerate(sampled_records):
        # For Manual Metrics
        m50 = evaluate_localization_manual(r["predictions"]["localization"], r["ground_truth"]["boxes"], 0.5)
        tp50 += m50["tp"]
        fp50 += m50["fp"]
        fn50 += m50["fn"]

        m30 = evaluate_localization_manual(r["predictions"]["localization"], r["ground_truth"]["boxes"], 0.3)
        tp30 += m30["tp"]
        fp30 += m30["fp"]
        fn30 += m30["fn"]

        # For COCO Metrics
        coco_gt["images"].append({"id": idx, "file_name": r["filename"]})
        for box in r["ground_truth"]["boxes"]:
          coco_gt["annotations"].append({
              "id": ann_id,
              "image_id": idx,
              "category_id": 1,
              "bbox": [box["x"], box["y"], box["width"], box["height"]],
              "area": box["width"] * box["height"],
              "iscrowd": 0,
          })
          ann_id += 1

        for pred in r["predictions"]["localization"]:
          box = pred.get("bbox_2d")
          if not box:
            continue
          ymin, xmin, ymax, xmax = box[0], box[1], box[2], box[3]
          x1 = min(xmin, xmax)
          x2 = max(xmin, xmax)
          y1 = min(ymin, ymax)
          y2 = max(ymin, ymax)
          coco_dt.append({
              "image_id": idx,
              "category_id": 1,
              "bbox": [x1, y1, x2 - x1, y2 - y1],
              "score": 1.0,
          })

      metrics["ACC50"] = tp50 / (tp50 + fp50 + fn50) if (tp50 + fp50 + fn50) > 0 else 0.0
      metrics["FNR"] = fn50 / (tp50 + fn50) if (tp50 + fn50) > 0 else 0.0

      p30 = tp30 / (tp30 + fp30) if (tp30 + fp30) > 0 else 0.0
      r30 = tp30 / (tp30 + fn30) if (tp30 + fn30) > 0 else 0.0
      metrics["mAP@30"] = 2 * p30 * r30 / (p30 + r30) if (p30 + r30) > 0 else 0.0

      # Run COCO evaluation
      temp_coco_gt_path = os.path.join(OUTPUT_DIR, f"temp_coco_gt_{i}.json")
      with open(temp_coco_gt_path, "w") as f:
        json.dump(coco_gt, f)

      try:
        coco = COCO(temp_coco_gt_path)
        coco_dt_obj = coco.loadRes(coco_dt) if coco_dt else COCO()
        coco_eval = COCOeval(coco, coco_dt_obj, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        metrics["mAP@[50:95]"] = float(coco_eval.stats[0])
        metrics["mAP@50"] = float(coco_eval.stats[1])
        metrics["mAP@75"] = float(coco_eval.stats[2])
      except Exception as e:
        print(f"COCO evaluation failed at iteration {i}: {e}")
        metrics["mAP@[50:95]"] = 0.0
        metrics["mAP@50"] = 0.0
        metrics["mAP@75"] = 0.0
      finally:
        if os.path.exists(temp_coco_gt_path):
          os.remove(temp_coco_gt_path)

    # 2. Captioning Metrics (BLEU-4, METEOR, F1s)
    if "BLEU-4" not in metrics:
      all_preds = []
      all_refs = []
      for r in sampled_records:
        pred = r["predictions"]["caption"]
        ref = r["ground_truth"]["caption"]
        
        if not isinstance(pred, str) and (pred is None or pred != pred):
          pred = "N/A"
        if not isinstance(ref, str) and (ref is None or ref != ref):
          ref = "N/A"
          
        all_preds.append(str(pred))
        all_refs.append([str(ref)])
      
      bleu_res = bleu_metric.compute(predictions=all_preds, references=all_refs)
      meteor_res = meteor_metric.compute(predictions=all_preds, references=all_refs)
      
      metrics["BLEU-4"] = bleu_res["bleu"]
      metrics["METEOR"] = meteor_res["meteor"]

      all_gt_keys = []
      all_pred_keys = []
      for r in sampled_records:
        all_gt_keys.extend(r["evaluation"]["caption"]["Standardized_Keywords"]["GT"])
        all_pred_keys.extend(r["evaluation"]["caption"]["Standardized_Keywords"]["Pred"])

      gt_set = set(all_gt_keys)
      pred_set = set(all_pred_keys)

      gt_modality = gt_set.intersection(modality_terms)
      pred_modality = pred_set.intersection(modality_terms)
      metrics["Modality_Term_F1"] = compute_f1(gt_modality, pred_modality)

      gt_clinical = gt_set.difference(modality_terms)
      pred_clinical = pred_set.difference(modality_terms)
      metrics["Clinical_Term_F1"] = compute_f1(gt_clinical, pred_clinical)

      gt_binary = [1 if r["evaluation"]["caption"]["Consistency"]["GT"] == "abnormal" else 0 for r in sampled_records]
      pred_binary = [1 if r["evaluation"]["caption"]["Consistency"]["Pred"] == "abnormal" else 0 for r in sampled_records]
      metrics["Binary_F1"] = compute_macro_f1(gt_binary, pred_binary)

    # 3. Reasoning Metrics
    if "Top-1 Accuracy" not in metrics:
      top1_correct = sum(1 for r in sampled_records if r["evaluation"]["reasoning"].get("Top_1") == "Correct")
      top5_correct = sum(1 for r in sampled_records if r["evaluation"]["reasoning"].get("Top_5") == "Correct")
      
      metrics["Top-1 Accuracy"] = top1_correct / len(sampled_records)
      metrics["Top-5 Accuracy"] = top5_correct / len(sampled_records)

    res_obj = {
        "iteration": i,
        "sampled_indices": sampled_indices,
        "metrics": metrics
    }
    new_results.append(res_obj)
    
    if i >= len(existing_results):
      with open(BOOTSTRAP_PATH, "a") as f:
        f.write(json.dumps(res_obj) + "\n")

  # Calculate and print CI
  metrics_history = {}
  for res in new_results:
    for k, v in res["metrics"].items():
      if k not in metrics_history:
        metrics_history[k] = []
      metrics_history[k].append(v)

  print("\n--- Bootstrap Results (95% Confidence Intervals) ---")
  for metric, values in metrics_history.items():
    mean_val = sum(values) / len(values)
    sorted_vals = sorted(values)
    lower_bound = sorted_vals[int(len(sorted_vals) * 0.025)]
    upper_bound = sorted_vals[int(len(sorted_vals) * 0.975)]
    print(f"{metric}: Mean={mean_val:.4f}, 95% CI=[{lower_bound:.4f}, {upper_bound:.4f}]")

if __name__ == "__main__":
  main()
