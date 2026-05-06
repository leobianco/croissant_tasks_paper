import base64
import collections
import io
import json
import os
from typing import Any, Dict, List
import datasets
import evaluate
import nltk
import numpy as np
import pandas as pd
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import requests

# Ensure API key is available
if "GEMINI_API_KEY" not in os.environ:
  raise ValueError("GEMINI_API_KEY environment variable not set.")

API_KEY = os.environ["GEMINI_API_KEY"]
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

# Download nltk resources if needed
try:
  nltk.data.find("tokenizers/punkt")
except LookupError:
  nltk.download("punkt")


def encode_image(image: Image.Image) -> str:
  """Encodes a PIL image to base64 string."""
  buffered = io.BytesIO()
  if image.mode != "RGB":
    image = image.convert("RGB")
  image.save(buffered, format="JPEG")
  return base64.b64encode(buffered.getvalue()).decode("utf-8")


def call_gemini(
    prompt: str, image: Image.Image = None, system_prompt: str = None
) -> str:
  """Calls Gemini API via REST with retry logic."""
  parts = []
  if image:
    parts.append(
        {"inlineData": {"mimeType": "image/jpeg", "data": encode_image(image)}}
    )
  parts.append({"text": prompt})

  contents = [{"parts": parts}]

  payload = {
      "contents": contents,
      "generationConfig": {"temperature": 0.1, "maxOutputTokens": 2048},
  }

  if system_prompt:
    payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

  max_retries = 50  # Heavily increased retries to insist after 503 errors
  for attempt in range(max_retries):
    response = requests.post(
        f"{API_URL}?key={API_KEY}",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=60,
    )

    if response.status_code == 200:
      break
    elif response.status_code == 503 and attempt < max_retries - 1:
      delay = 2  # Fixed small delay
      print(f"503 error, retrying in {delay} seconds...")
      import time

      time.sleep(delay)
    else:
      raise RuntimeError(f"API call failed: {response.text}")

  try:
    result = response.json()
    return result["candidates"][0]["content"]["parts"][0]["text"]
  except (KeyError, IndexError) as e:
    raise RuntimeError(f"Failed to parse API response: {response.text}") from e


# --- Evaluation Metrics Implementation ---


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


def evaluate_localization_manual(
    preds: list[Dict[str, Any]], golds: list[Dict[str, Any]], iou_threshold=0.5
) -> Dict[str, float]:
  """Computes Precision, Recall, and F1 for localization at a given IoU threshold."""
  if not golds:
    return {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
    }

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
      gold_box = [
          gold["y"],
          gold["x"],
          gold["y"] + gold["height"],
          gold["x"] + gold["width"],
      ]
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
  f1 = (
      2 * precision * recall / (precision + recall)
      if (precision + recall) > 0
      else 0.0
  )

  return {
      "precision": precision,
      "recall": recall,
      "f1": f1,
      "tp": tp,
      "fp": fp,
      "fn": fn,
  }


def compute_f1(gt_set, pred_set):
  """Computes F1 score between two sets of items."""
  intersection = gt_set.intersection(pred_set)
  precision = len(intersection) / len(pred_set) if len(pred_set) > 0 else 0.0
  recall = len(intersection) / len(gt_set) if len(gt_set) > 0 else 0.0
  return (
      2 * precision * recall / (precision + recall)
      if (precision + recall) > 0
      else 0.0
  )


# --- Task Runners ---


def run_localization(image: Image.Image) -> Dict[str, Any]:
  """Task 1: Anomaly Localization (Plain Text Output)."""
  prompt = (
      "Return bounding boxes of any abnormal areas in the following format,"
      " one per line:\n"
      "[label]: [ymin, xmin, ymax, xmax]\n\n"
      "Use normalized coordinates (0 to 1000) for the image dimensions.\n"
      'If no abnormality is found, return "no target".\n\n'
      "Example:\n"
      "hydrocephalus: [250, 250, 749, 800]"
  )

  response_text = call_gemini(prompt, image=image)
  lines = response_text.strip().split("\n")
  preds = []

  for line in lines:
    if ":" in line and "[" in line and "]" in line:
      try:
        label = line.split(":", 1)[0].strip()
        coords_str = line.split(":", 1)[1].strip().strip("[]")
        coords = [int(c.strip()) for c in coords_str.split(",")]
        if len(coords) == 4:
          preds.append({"box_2d": coords, "label": label})
      except Exception as e:
        print(f"Failed to parse line: {line}. Error: {e}")

  return {"raw_output": response_text, "parsed": preds}


def run_captioning(image: Image.Image) -> Dict[str, Any]:
  """Task 2: Image Description."""
  system_prompt = (
      "You are a highly skilled radiologist AI assistant. Your task is to"
      " analyze medical images with precision and generate accurate, concise"
      " diagnostic descriptions suitable for clinical use. Always prioritize"
      " clarity, accuracy, and domain-specific terminology in your responses."
  )

  prompt = (
      "Please carefully examine the provided medical image and perform a"
      " comprehensive, in-depth analysis. Generate a clear, concise description"
      " focusing on the imaging modality, slice orientation, lesion location,"
      " and any notable abnormalities observed.\n"
      "Format to Follow:\n"
      "- Answer:\n"
      "[Only output the final concise description result.]"
  )

  response_text = call_gemini(prompt, image=image, system_prompt=system_prompt)
  return {"raw_output": response_text}


def run_reasoning(clinical_history: str, mri_findings: str) -> Dict[str, Any]:
  """Task 3: Diagnostic Reasoning."""
  prompt = (
      "Please provide the most likely diagnosis followed by four other possible"
      " differential diagnoses based on the following clinical history and MRI"
      " findings.\n\n"
      f'Clinical History:\n"{clinical_history}"\n\n'
      f'MRI Findings:\n"{mri_findings}"\n\n'
      "Output Format:\n"
      "1. Most likely diagnosis: [Diagnosis]\n"
      "2. Alternative 1: [Diagnosis]\n"
      "3. Alternative 2: [Diagnosis]\n"
      "4. Alternative 3: [Diagnosis]\n"
      "5. Alternative 4: [Diagnosis]"
  )

  response_text = call_gemini(prompt)
  lines = response_text.strip().split("\n")
  most_likely = "N/A"
  alternatives = []

  for line in lines:
    if "1. Most likely diagnosis:" in line:
      most_likely = line.split(":", 1)[1].strip().strip("[]")
    elif any(f"{i}. Alternative" in line for i in range(2, 6)):
      if ":" in line:
        alt = line.split(":", 1)[1].strip().strip("[]")
        alternatives.append(alt)
      else:
        parts = line.split("-", 1)
        if len(parts) > 1:
          alt = parts[1].strip().strip("[]")
          alternatives.append(alt)
        else:
          print(f"Warning: Unexpected line format in reasoning output: {line}")

  parsed = {
      "most_likely_diagnosis": most_likely,
      "other_possible_diagnoses": alternatives,
  }
  return {"raw_output": response_text, "parsed": parsed}


# --- LLM Evaluators ---


def evaluate_caption_llm(gt: str, pred: str) -> Dict[str, Any]:
  """Uses Gemini to evaluate captions."""
  prompt = f"""You are given two radiology reports: Ground Truth (GT) and Predicted (Pred). Your task is to extract and standardize medically important keywords from both reports.

Task: Extract keywords related to the following categories:
• Anatomical structures: e.g., brain regions, body parts.
• Imaging characteristics: e.g., hyperintensity, low density, enhancement, mass-like, signal changes.
• Disease or pathological findings: e.g., leukoencephalopathy, infarct, tumor.
• Negated findings: any finding explicitly stated as absent or negative, such as “no hemorrhage”, “no mass” — keep the negation in the keyword.
• Imaging sequence and plane: e.g., T1, T2, FLAIR, DWI, sagittal, axial, coronal.

Standardization Rules:
• Normalize synonymous or semantically similar expressions into a single canonical form.
• Normalize anatomical mentions related to disease into their broader anatomical structures when appropriate.
• Ensure that after normalization, all terms that refer to the same concept are exactly string-equal, to support direct set-based comparison.

Output Format:
Output exactly two lines:
GT: [comma separated list of standardized keywords]
Pred: [comma separated list of standardized keywords]
Consistency: [GT_status] | [Pred_status] | [true/false] (where status is 'normal' or 'abnormal')

Input:
GT = "{gt}"
Pred = "{pred}"
"""
  response_text = call_gemini(prompt)
  lines = response_text.strip().split("\n")
  gt_keys = []
  pred_keys = []
  consistency = {"Is_Consistent": False, "GT": "normal", "Pred": "normal"}

  for line in lines:
    if line.startswith("GT:"):
      gt_keys = [k.strip() for k in line[3:].strip("[]").split(",")]
    elif line.startswith("Pred:"):
      pred_keys = [k.strip() for k in line[5:].strip("[]").split(",")]
    elif line.startswith("Consistency:"):
      parts = line[12:].split("|")
      if len(parts) == 3:
        consistency["GT"] = parts[0].strip()
        consistency["Pred"] = parts[1].strip()
        consistency["Is_Consistent"] = parts[2].strip().lower() == "true"

  return {
      "Standardized_Keywords": {"GT": gt_keys, "Pred": pred_keys},
      "Consistency": consistency,
  }


def evaluate_diagnosis_llm(gt: str, pred: Dict[str, Any]) -> Dict[str, Any]:
  """Uses Gemini to evaluate diagnosis."""
  pred_str = json.dumps(pred)
  prompt = f"""You are a professional medical diagnosis evaluation system. You will receive two inputs:
• Ground Truth Diagnosis (GT): A single confirmed diagnosis.
• Predicted Diagnosis (Pred): One most likely diagnosis and four additional possible diagnosis candidates.

Evaluation Rules
• Focus only on the core diagnosis, regardless of etiology or cause.
• Allow for synonyms and variations in medical terminology.
• If the same diagnostic entity (imaging pattern, pathological finding, or clinical condition) is present in the predictions, consider it correct.
• Do not penalize for differences in specificity or cause (e.g., idiopathic vs secondary), unless the disease is fundamentally different.

Input:
GT: "{gt}"
Pred: "{pred_str}"

Output Format:
Return only JSON in the following structure:
{{
"Top_1": "Correct" | "Wrong",
"Reason_for_Top1": "<your explanation>",
"Top_5": "Correct" | "Wrong",
"Reason_for_Top5": "<your explanation>"
}}
Only return valid JSON with no extra text.
"""
  response_text = call_gemini(prompt)
  if "```json" in response_text:
    response_text = response_text.split("```json")[1].split("```")[0].strip()
  try:
    return json.loads(response_text)
  except json.JSONDecodeError:
    return {}


# --- Main ---


def main():
  print("Loading ground truth files...")
  try:
    golds_bbox = pd.read_csv("nova_results/data/bboxes_gold.csv")
    golds_caption = pd.read_csv("nova_results/data/captions.csv")
    metadata = pd.read_csv("nova_results/data/case_metadata.csv")
  except Exception as e:
    print(f"Failed to load ground truth files: {e}")
    return

  print("Loading dataset c-i-ber/Nova...")
  try:
    dataset = datasets.load_dataset("c-i-ber/Nova")
    data = dataset["train"] if "train" in dataset else dataset
  except Exception as e:
    print(f"Failed to load dataset: {e}")
    return

  # Incremental Execution Setup
  output_dir = "nova_results/implementation/outputs"
  os.makedirs(output_dir, exist_ok=True)
  results_path = os.path.join(output_dir, "full_results.json")

  results = []
  if os.path.exists(results_path):
    print(f"Loading existing results from {results_path}...")
    try:
      with open(results_path, "r", encoding="utf-8") as f:
        saved_data = json.load(f)
        results = saved_data.get("results", [])
    except Exception as e:
      print(f"Failed to load existing results: {e}")

  processed_filenames = {r["filename"] for r in results}
  unprocessed_data = [
      d
      for d in data
      if os.path.basename(d["image"].filename) not in processed_filenames
  ]

  limit = 0  # Only compute metrics on existing results
  num_samples = min(limit, len(unprocessed_data))
  print(f"Found {len(results)} already processed samples.")
  print(
      f"Running evaluation on {num_samples} NEW samples out of"
      f" {len(unprocessed_data)} unprocessed."
  )

  if num_samples == 0 and not results:
    print("No new samples to process and no existing results.")
    return

  bleu_metric = evaluate.load("bleu")
  meteor_metric = evaluate.load("meteor")

  modality_terms = {
      "flair",
      "axial",
      "sagittal",
      "t1",
      "t2",
      "coronal",
      "dwi",
      "t1w",
      "t2w",
      "weighted",
  }

  for i in range(num_samples):
    sample = unprocessed_data[i]
    image = sample.get("image")
    width, height = image.size
    filename = os.path.basename(image.filename)
    print(f"\nProcessing sample {i+1}/{num_samples}: {filename}")
    print(f"Image size: {width}x{height}")

    # Lookup Ground Truth
    gt_boxes = golds_bbox[golds_bbox["filename"] == filename].to_dict("records")
    gt_caption_row = golds_caption[golds_caption["filename"] == filename]
    gt_caption = (
        gt_caption_row["caption"].values[0]
        if not gt_caption_row.empty
        else "N/A"
    )
    case_id = (
        gt_caption_row["case_id"].values[0]
        if not gt_caption_row.empty
        else "N/A"
    )

    gt_metadata = metadata[metadata["case_id"] == case_id]
    clinical_history = (
        gt_metadata["clinical_history"].values[0]
        if not gt_metadata.empty
        else "N/A"
    )
    final_diagnosis = (
        gt_metadata["final_diagnosis"].values[0]
        if not gt_metadata.empty
        else "N/A"
    )

    # Run Predictions
    loc_res = run_localization(image)
    cap_res = run_captioning(image)
    reas_res = run_reasoning(clinical_history, gt_caption)

    # Scale predictions from 0-1000 to pixel coordinates
    scaled_preds = []
    for pred in loc_res["parsed"]:
      box = pred.get("box_2d")
      if not box or len(box) != 4:
        continue
      ymin_norm, xmin_norm, ymax_norm, xmax_norm = (
          box[0],
          box[1],
          box[2],
          box[3],
      )

      ymin_pixel = ymin_norm * height / 1000.0
      xmin_pixel = xmin_norm * width / 1000.0
      ymax_pixel = ymax_norm * height / 1000.0
      xmax_pixel = xmax_norm * width / 1000.0

      scaled_preds.append({
          "bbox_2d": [ymin_pixel, xmin_pixel, ymax_pixel, xmax_pixel],
          "label": pred.get("label", "anomaly"),
      })

    # Evaluate Captioning & Reasoning per sample
    cap_eval = evaluate_caption_llm(gt_caption, cap_res["raw_output"])
    reas_eval = evaluate_diagnosis_llm(final_diagnosis, reas_res["parsed"])

    # Append to results immediately
    results.append({
        "filename": filename,
        "predictions": {
            "localization": scaled_preds,
            "caption": cap_res["raw_output"],
            "diagnosis": reas_res["parsed"],
        },
        "ground_truth": {
            "boxes": gt_boxes,
            "caption": gt_caption,
            "diagnosis": final_diagnosis,
        },
        "evaluation": {"caption": cap_eval, "reasoning": reas_eval},
    })

    # Save periodically (every 10 samples)
    if (i + 1) % 10 == 0 or (i + 1) == num_samples:
      print(f"Saving intermediate results ({i+1}/{num_samples})...")
      with open(results_path, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2)

  # --- Aggregate Evaluation on ALL Results ---
  print("\nComputing aggregate metrics on all processed samples...")
  metrics_summary = {"localization": {}, "captioning": {}, "reasoning": {}}

  # 1. Localization
  coco_gt = {
      "images": [],
      "annotations": [],
      "categories": [{"id": 1, "name": "anomaly"}],
  }
  coco_dt = []
  ann_id = 1

  for idx, r in enumerate(results):
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

  temp_coco_gt_path = os.path.join(output_dir, "coco_gt_temp.json")
  with open(temp_coco_gt_path, "w") as f:
    json.dump(coco_gt, f)

  try:
    coco = COCO(temp_coco_gt_path)
    coco_dt_obj = coco.loadRes(coco_dt) if coco_dt else COCO()
    coco_eval = COCOeval(coco, coco_dt_obj, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    coco_eval_30 = COCOeval(coco, coco_dt_obj, "bbox")
    import numpy as np

    coco_eval_30.params.iouThrs = np.array([0.3])
    coco_eval_30.evaluate()
    coco_eval_30.accumulate()
    coco_eval_30.summarize()

    metrics_summary["localization"] = {
        "mAP@[50:95]": float(coco_eval.stats[0]),
        "mAP@50": float(coco_eval.stats[1]),
        "mAP@75": float(coco_eval.stats[2]),
        "mAP@30": float(coco_eval_30.stats[0]),
    }
  except Exception as e:
    print(f"COCO evaluation failed: {e}")
    metrics_summary["localization"]["error"] = str(e)

  tp50, fp50, fn50 = 0, 0, 0
  tp30, fp30, fn30 = 0, 0, 0

  for r in results:
    m50 = evaluate_localization_manual(
        r["predictions"]["localization"], r["ground_truth"]["boxes"], 0.5
    )
    tp50 += m50["tp"]
    fp50 += m50["fp"]
    fn50 += m50["fn"]

    m30 = evaluate_localization_manual(
        r["predictions"]["localization"], r["ground_truth"]["boxes"], 0.3
    )
    tp30 += m30["tp"]
    fp30 += m30["fp"]
    fn30 += m30["fn"]

  p30 = tp30 / (tp30 + fp30) if (tp30 + fp30) > 0 else 0.0
  r30 = tp30 / (tp30 + fn30) if (tp30 + fn30) > 0 else 0.0
  metrics_summary["localization"]["mAP@30"] = (
      2 * p30 * r30 / (p30 + r30) if (p30 + r30) > 0 else 0.0
  )

  metrics_summary["localization"]["ACC50"] = (
      tp50 / (tp50 + fp50 + fn50) if (tp50 + fp50 + fn50) > 0 else 0.0
  )
  metrics_summary["localization"]["TP30"] = tp30
  metrics_summary["localization"]["FP30"] = fp30
  metrics_summary["localization"]["FNR"] = (
      fn50 / (tp50 + fn50) if (tp50 + fn50) > 0 else 0.0
  )

  # 2. Captioning
  all_preds = [r["predictions"]["caption"] for r in results]
  all_refs = [[r["ground_truth"]["caption"]] for r in results]

  bleu_res = bleu_metric.compute(predictions=all_preds, references=all_refs)
  meteor_res = meteor_metric.compute(predictions=all_preds, references=all_refs)

  metrics_summary["captioning"] = {
      "BLEU-4": bleu_res["bleu"],
      "METEOR": meteor_res["meteor"],
  }

  all_gt_keys = []
  all_pred_keys = []
  for r in results:
    all_gt_keys.extend(
        r["evaluation"]["caption"]["Standardized_Keywords"]["GT"]
    )
    all_pred_keys.extend(
        r["evaluation"]["caption"]["Standardized_Keywords"]["Pred"]
    )

  gt_set = set(all_gt_keys)
  pred_set = set(all_pred_keys)

  gt_modality = gt_set.intersection(modality_terms)
  pred_modality = pred_set.intersection(modality_terms)
  metrics_summary["captioning"]["Modality_Term_F1"] = compute_f1(
      gt_modality, pred_modality
  )

  gt_clinical = gt_set.difference(modality_terms)
  pred_clinical = pred_set.difference(modality_terms)
  metrics_summary["captioning"]["Clinical_Term_F1"] = compute_f1(
      gt_clinical, pred_clinical
  )

  gt_binary = [
      1 if r["evaluation"]["caption"]["Consistency"]["GT"] == "abnormal" else 0
      for r in results
  ]
  pred_binary = [
      1
      if r["evaluation"]["caption"]["Consistency"]["Pred"] == "abnormal"
      else 0
      for r in results
  ]

  def compute_macro_f1(y_true, y_pred):
    def f1_for_class(cls):
      tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
      fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
      fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
      precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
      recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
      return (
          2 * precision * recall / (precision + recall)
          if (precision + recall) > 0
          else 0.0
      )

    return (f1_for_class(0) + f1_for_class(1)) / 2

  metrics_summary["captioning"]["Binary_F1"] = compute_macro_f1(
      gt_binary, pred_binary
  )

  # 3. Reasoning
  top1_correct = sum(
      1
      for r in results
      if r["evaluation"]["reasoning"].get("Top_1") == "Correct"
  )
  top5_correct = sum(
      1
      for r in results
      if r["evaluation"]["reasoning"].get("Top_5") == "Correct"
  )

  metrics_summary["reasoning"] = {
      "Top-1 Accuracy": top1_correct / len(results),
      "Top-5 Accuracy": top5_correct / len(results),
  }

  print("\n--- Final Metrics Summary ---")
  print(json.dumps(metrics_summary, indent=2))

  # Save combined results with summary
  with open(results_path, "w", encoding="utf-8") as f:
    json.dump({"results": results, "summary": metrics_summary}, f, indent=2)
  print(f"Combined results saved to {results_path}")

  # Save separate outputs per sub-task
  loc_outputs = [
      {
          "filename": r["filename"],
          "predictions": r["predictions"]["localization"],
      }
      for r in results
  ]
  with open(os.path.join(output_dir, "localization_preds.json"), "w") as f:
    json.dump(loc_outputs, f, indent=2)

  cap_outputs = [
      {"filename": r["filename"], "predictions": r["predictions"]["caption"]}
      for r in results
  ]
  with open(os.path.join(output_dir, "captioning_preds.json"), "w") as f:
    json.dump(cap_outputs, f, indent=2)

  reas_outputs = [
      {"filename": r["filename"], "predictions": r["predictions"]["diagnosis"]}
      for r in results
  ]
  with open(os.path.join(output_dir, "reasoning_preds.json"), "w") as f:
    json.dump(reas_outputs, f, indent=2)

  # Generate TaskSolution
  impl_url = "./evaluate_baseline.py"

  solution = {
      "@context": {
          "croissant": "http://mlcommons.org/croissant/",
          "schema": "https://schema.org/",
          "xsd": "http://www.w3.org/2001/XMLSchema#",
      },
      "@type": "croissant:TaskSolution",
      "@id": (
          "https://openreview.net/forum?id=DwFDfrPsm8_solution_gemini_2_5_flash"
      ),
      "schema:name": "NOVA Solution - Gemini 2.5 Flash",
      "schema:isBasedOn": {"@id": "https://openreview.net/forum?id=DwFDfrPsm8"},
      "croissant:subTask": [
          {
              "@type": "croissant:TaskSolution",
              "@id": (
                  "https://openreview.net/forum?id=DwFDfrPsm8_solution_gemini_2_5_flash#localization"
              ),
              "schema:name": "NOVA Solution - Anomaly Localization",
              "schema:isBasedOn": {
                  "@id": (
                      "https://openreview.net/forum?id=DwFDfrPsm8#localization"
                  )
              },
              "croissant:implementation": {
                  "@type": "schema:SoftwareApplication",
                  "@id": (
                      "https://openreview.net/forum?id=DwFDfrPsm8_solution_gemini_2_5_flash#localization_impl"
                  ),
                  "schema:name": "Gemini 2.5 Flash Localization Baseline",
                  "schema:url": impl_url,
              },
              "croissant:output": {
                  "@type": "schema:Dataset",
                  "@id": "urn:uuid:nova-localization-outputs",
                  "schema:name": "Localization Outputs",
                  "schema:url": (
                      f"file://{os.path.abspath(os.path.join(output_dir, 'localization_preds.json'))}"
                  ),
              },
              "croissant:evaluation": {
                  "@type": "croissant:EvaluationTask",
                  "schema:name": "Evaluation of Localization",
                  "croissant:evaluationResults": [
                      {
                          "@type": "croissant:EvaluationResult",
                          "croissant:metric": "mAP@[50:95]",
                          "croissant:value": (
                              metrics_summary["localization"].get("mAP@[50:95]")
                          ),
                      },
                      {
                          "@type": "croissant:EvaluationResult",
                          "croissant:metric": "mAP@50",
                          "croissant:value": (
                              metrics_summary["localization"].get("mAP@50")
                          ),
                      },
                      {
                          "@type": "croissant:EvaluationResult",
                          "croissant:metric": "mAP@30",
                          "croissant:value": (
                              metrics_summary["localization"].get("mAP@30")
                          ),
                      },
                      {
                          "@type": "croissant:EvaluationResult",
                          "croissant:metric": "ACC50",
                          "croissant:value": (
                              metrics_summary["localization"].get("ACC50")
                          ),
                      },
                      {
                          "@type": "croissant:EvaluationResult",
                          "croissant:metric": "TP30",
                          "croissant:value": (
                              metrics_summary["localization"].get("TP30")
                          ),
                      },
                      {
                          "@type": "croissant:EvaluationResult",
                          "croissant:metric": "FP30",
                          "croissant:value": (
                              metrics_summary["localization"].get("FP30")
                          ),
                      },
                      {
                          "@type": "croissant:EvaluationResult",
                          "croissant:metric": "FNR",
                          "croissant:value": (
                              metrics_summary["localization"].get("FNR")
                          ),
                      },
                  ],
              },
          },
          {
              "@type": "croissant:TaskSolution",
              "@id": (
                  "https://openreview.net/forum?id=DwFDfrPsm8_solution_gemini_2_5_flash#captioning"
              ),
              "schema:name": "NOVA Solution - Image Description",
              "schema:isBasedOn": {
                  "@id": "https://openreview.net/forum?id=DwFDfrPsm8#captioning"
              },
              "croissant:implementation": {
                  "@type": "schema:SoftwareApplication",
                  "@id": (
                      "https://openreview.net/forum?id=DwFDfrPsm8_solution_gemini_2_5_flash#captioning_impl"
                  ),
                  "schema:name": "Gemini 2.5 Flash Captioning Baseline",
                  "schema:url": impl_url,
              },
              "croissant:output": {
                  "@type": "schema:Dataset",
                  "@id": "urn:uuid:nova-captioning-outputs",
                  "schema:name": "Captioning Outputs",
                  "schema:url": (
                      f"file://{os.path.abspath(os.path.join(output_dir, 'captioning_preds.json'))}"
                  ),
              },
              "croissant:evaluation": {
                  "@type": "croissant:EvaluationTask",
                  "schema:name": "Evaluation of Captioning",
                  "croissant:evaluationResults": [
                      {
                          "@type": "croissant:EvaluationResult",
                          "croissant:metric": "BLEU-4",
                          "croissant:value": (
                              metrics_summary["captioning"].get("BLEU-4")
                          ),
                      },
                      {
                          "@type": "croissant:EvaluationResult",
                          "croissant:metric": "METEOR",
                          "croissant:value": (
                              metrics_summary["captioning"].get("METEOR")
                          ),
                      },
                      {
                          "@type": "croissant:EvaluationResult",
                          "croissant:metric": "Clinical_Term_F1",
                          "croissant:value": (
                              metrics_summary["captioning"].get(
                                  "Clinical_Term_F1"
                              )
                          ),
                      },
                      {
                          "@type": "croissant:EvaluationResult",
                          "croissant:metric": "Modality_Term_F1",
                          "croissant:value": (
                              metrics_summary["captioning"].get(
                                  "Modality_Term_F1"
                              )
                          ),
                      },
                      {
                          "@type": "croissant:EvaluationResult",
                          "croissant:metric": "Binary_F1",
                          "croissant:value": (
                              metrics_summary["captioning"].get("Binary_F1")
                          ),
                      },
                  ],
              },
          },
          {
              "@type": "croissant:TaskSolution",
              "@id": (
                  "https://openreview.net/forum?id=DwFDfrPsm8_solution_gemini_2_5_flash#reasoning"
              ),
              "schema:name": "NOVA Solution - Diagnostic Reasoning",
              "schema:isBasedOn": {
                  "@id": "https://openreview.net/forum?id=DwFDfrPsm8#reasoning"
              },
              "croissant:implementation": {
                  "@type": "schema:SoftwareApplication",
                  "@id": (
                      "https://openreview.net/forum?id=DwFDfrPsm8_solution_gemini_2_5_flash#reasoning_impl"
                  ),
                  "schema:name": "Gemini 2.5 Flash Reasoning Baseline",
                  "schema:url": impl_url,
              },
              "croissant:output": {
                  "@type": "schema:Dataset",
                  "@id": "urn:uuid:nova-reasoning-outputs",
                  "schema:name": "Reasoning Outputs",
                  "schema:url": (
                      f"file://{os.path.abspath(os.path.join(output_dir, 'reasoning_preds.json'))}"
                  ),
              },
              "croissant:evaluation": {
                  "@type": "croissant:EvaluationTask",
                  "schema:name": "Evaluation of Reasoning",
                  "croissant:evaluationResults": [
                      {
                          "@type": "croissant:EvaluationResult",
                          "croissant:metric": "Top-1 Accuracy",
                          "croissant:value": metrics_summary["reasoning"][
                              "Top-1 Accuracy"
                          ],
                      },
                      {
                          "@type": "croissant:EvaluationResult",
                          "croissant:metric": "Top-5 Accuracy",
                          "croissant:value": metrics_summary["reasoning"][
                              "Top-5 Accuracy"
                          ],
                      },
                  ],
              },
          },
      ],
  }

  solution_path = "nova_results/solutions/gemini_2_5_flash.jsonld"
  with open(solution_path, "w", encoding="utf-8") as f:
    json.dump(solution, f, indent=2)
  print(f"TaskSolution file generated at {solution_path}")


if __name__ == "__main__":
  main()
