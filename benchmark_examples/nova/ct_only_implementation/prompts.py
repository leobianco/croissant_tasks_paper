"""Prompts for Gemini baseline on the NOVA benchmark."""

ABNORMALITY_GROUNDING_PROMPT = """Return bounding boxes of any abnormal areas in the following format, one per line:
[label]: [ymin, xmin, ymax, xmax]

Use normalized coordinates (0 to 1000) for the image dimensions.
If no abnormality is found, return "no target".

Example:
hydrocephalus: [250, 250, 749, 800]"""


IMAGE_CAPTIONING_PROMPT = """System Prompt:
You are a highly skilled radiologist AI assistant. Your task is to analyze medical images with precision and generate accurate, concise diagnostic descriptions suitable for clinical use. Always prioritize clarity, accuracy, and domain-specific terminology in your responses.

Please carefully examine the provided medical image and perform a comprehensive, in-depth analysis.
Generate a clear, concise description focusing on the imaging modality, slice orientation, lesion location, and any notable abnormalities observed.

Format to Follow:
- Answer:
[Only output the final concise description result.]"""

DIFFERENTIAL_DIAGNOSIS_PROMPT = """Please provide the most likely diagnosis along with four other possible differential diagnoses based on the following clinical history and MRI findings. Your output should be structured in JSON format.

Clinical History:
"{clinical_history}"

MRI Findings:
"{mri_findings}"

Format to Follow:
json
{{
  "most_likely_diagnosis": "Diagnosis name here",
  "other_possible_diagnoses": [
    "Diagnosis 1 here",
    "Diagnosis 2 here",
    "Diagnosis 3 here",
    "Diagnosis 4 here"
  ]
}}"""



IMAGE_DESCRIPTION_EVALUATION_PROMPT = """You are given two radiology reports: Ground Truth (GT) and Predicted (Pred). Your task is to extract and
standardize medically important keywords from both reports.Task: Extract keywords related to the following categories:
• Anatomical structures: e.g., brain regions, body parts.
• Imaging characteristics: e.g., hyperintensity, low density, enhancement, mass-like, signal changes.
• Disease or pathological findings: e.g., leukoencephalopathy, infarct, tumor.
• Negated findings: any finding explicitly stated as absent or negative, such as “no hemorrhage”, “no
mass” — keep the negation in the keyword.
• Imaging sequence and plane: e.g., T1, T2, FLAIR, DWI, sagittal, axial, coronal.Standardization Rules:• Normalize synonymous or semantically similar expressions into a single canonical form.
• Normalize anatomical mentions related to disease into their broader anatomical structures when
appropriate.
• Ensure that after normalization, all terms that refer to the same concept are exactly string-equal, to
support direct set-based comparison (e.g., for intersection/union using string matching).
• Prefer higher-level or broader terms when multiple expressions refer to variations of the same
anatomical area (e.g., “inferior pointing of the ventricles”, “ventricles slightly enlarged”, and
“ventricular dilation” should all be normalized to “ventricles”).
• The goal is to eliminate variation in expression and granularity, so that conceptually equivalent
phrases normalize to the same string.Consistency• GT and Pred are labeled as “normal” or “abnormal” based on their findings.
• Is_Consistent is true if both GT and Pred are either “normal” or both “abnormal”.
• Is_Consistent is false if one is “normal” and the other is “abnormal”.Input:GT = "{gt_input}"Pred = "{pred_input}"Output Format (JSON):{{
"Raw_Keywords": {{
"GT": ["keyword1", "keyword2", "..."],
"Pred": ["keyword1", "keyword2", "..."]
}},
"Standardized_Keywords": {{
"GT": ["standardized_keyword1", "standardized_keyword2", "..."],
"Pred": ["standardized_keyword1", "standardized_keyword2", "..."]
}},
"Consistency": {{
"GT": "normal" | "abnormal",
"Pred": "normal" | "abnormal",
"Is_Consistent": true | false
}}
}}
Only return valid JSON with no extra text."""



MEDICAL_DIAGNOSIS_EVALUATION_PROMPT = """You are a professional medical diagnosis evaluation system. You will receive two inputs:
• Ground Truth Diagnosis (GT): A single confirmed diagnosis.
• Predicted Diagnosis (Pred): One most likely diagnosis and four additional possible diagnosis
candidates.Evaluation Rules• Focus only on the core diagnosis, regardless of etiology or cause.
• Allow for synonyms and variations in medical terminology.
• If the same diagnostic entity (imaging pattern, pathological finding, or clinical condition) is present
in the predictions, consider it correct.
• Do not penalize for differences in specificity or cause (e.g., idiopathic vs secondary), unless the
disease is fundamentally different.Input:GT: "{gt_diagnosis}"Pred: "{pred_diagnosis}"Output Format:Return only JSON in the following structure:{{
"Top_1": "Correct" | "Wrong",
"Reason_for_Top1": "<your explanation>",
"Top_5": "Correct" | "Wrong",
"Reason_for_Top5": "<your explanation>"
}}
Only return valid JSON with no extra text."""


