import json

def build_solution_jsonld(eval_metrics_dict):
    """Builds the TaskSolution JSON-LD for Gemini 2.5 Flash."""
    return {
        "@context": {
            "croissant": "http://mlcommons.org/croissant/",
            "schema": "https://schema.org/",
            "xsd": "http://www.w3.org/2001/XMLSchema#"
        },
        "@type": "croissant:TaskSolution",
        "@id": "https://huggingface.co/datasets/c-i-ber/Nova_solution_gemini_2_5_flash",
        "schema:name": "NOVA Solution - Gemini 2.5 Flash",
        "schema:isBasedOn": { "@id": "https://huggingface.co/datasets/c-i-ber/Nova" },
        "croissant:implementation": {
            "@type": "schema:SoftwareApplication",
            "@id": "https://huggingface.co/datasets/c-i-ber/Nova_solution_gemini_2_5_flash#implementation",
            "schema:name": "Gemini 2.5 Flash"
        },
        "croissant:execution": {
            "@type": "croissant:ExecutionConfig",
            "@id": "https://huggingface.co/datasets/c-i-ber/Nova_solution_gemini_2_5_flash#execution",
            "croissant:hyperparameter": [
                {
                    "@type": "schema:PropertyValue",
                    "schema:name": "temperature",
                    "schema:value": 0.1
                },
                {
                    "@type": "schema:PropertyValue",
                    "schema:name": "max_tokens",
                    "schema:value": 2048
                }
            ]
        },
        "croissant:output": {
            "@type": "schema:Dataset",
            "@id": "urn:uuid:nova-solution-gemini_2_5_flash-output",
            "schema:name": "NOVA Solution Outputs - Gemini 2.5 Flash"
        },
        "croissant:subTask": [
            {
                "@type": "croissant:TaskSolution",
                "@id": "https://huggingface.co/datasets/c-i-ber/Nova_solution_gemini_2_5_flash#anomaly_localization_sol",
                "schema:isBasedOn": { "@id": "https://huggingface.co/datasets/c-i-ber/Nova#anomaly_localization" },
                "croissant:execution": { "@id": "https://huggingface.co/datasets/c-i-ber/Nova_solution_gemini_2_5_flash#execution" },
                "croissant:implementation": { "@id": "https://huggingface.co/datasets/c-i-ber/Nova_solution_gemini_2_5_flash#implementation" },
                "croissant:output": {
                    "@type": "schema:Dataset",
                    "@id": "urn:uuid:nova-solution-gemini_2_5_flash-anomaly_localization-output"
                },
                "croissant:evaluation": {
                    "@type": "croissant:EvaluationTask",
                    "@id": "https://huggingface.co/datasets/c-i-ber/Nova_evaluation_gemini_2_5_flash#anomaly_localization",
                    "schema:isBasedOn": { "@id": "https://huggingface.co/datasets/c-i-ber/Nova#anomaly_localization" },
                    "schema:name": "Task 1: Anomaly Localization",
                    "croissant:evaluatedSolution": { "@id": "https://huggingface.co/datasets/c-i-ber/Nova_solution_gemini_2_5_flash#anomaly_localization_sol" },
                    "croissant:evaluationResults": [
                        { "@type": "croissant:EvaluationResult", "croissant:metric": "mAP@30", "croissant:value": str(eval_metrics_dict.get("mAP@30", 0.0)) },
                        { "@type": "croissant:EvaluationResult", "croissant:metric": "mAP@50", "croissant:value": str(eval_metrics_dict.get("mAP@50", 0.0)) },
                        { "@type": "croissant:EvaluationResult", "croissant:metric": "mAP@50-95", "croissant:value": str(eval_metrics_dict.get("mAP@50-95", 0.0)) },
                        { "@type": "croissant:EvaluationResult", "croissant:metric": "ACC50", "croissant:value": str(eval_metrics_dict.get("ACC50", 0.0)) },
                        { "@type": "croissant:EvaluationResult", "croissant:metric": "TP30", "croissant:value": str(eval_metrics_dict.get("TP30", 0)) },
                        { "@type": "croissant:EvaluationResult", "croissant:metric": "FP30", "croissant:value": str(eval_metrics_dict.get("FP30", 0)) },
                        { "@type": "croissant:EvaluationResult", "croissant:metric": "FNR", "croissant:value": str(eval_metrics_dict.get("FNR", 0.0)) }
                    ]
                }
            },
            {
                "@type": "croissant:TaskSolution",
                "@id": "https://huggingface.co/datasets/c-i-ber/Nova_solution_gemini_2_5_flash#image_captioning_sol",
                "schema:isBasedOn": { "@id": "https://huggingface.co/datasets/c-i-ber/Nova#image_captioning" },
                "croissant:execution": { "@id": "https://huggingface.co/datasets/c-i-ber/Nova_solution_gemini_2_5_flash#execution" },
                "croissant:implementation": { "@id": "https://huggingface.co/datasets/c-i-ber/Nova_solution_gemini_2_5_flash#implementation" },
                "croissant:output": {
                    "@type": "schema:Dataset",
                    "@id": "urn:uuid:nova-solution-gemini_2_5_flash-image_captioning-output"
                },
                "croissant:evaluation": {
                    "@type": "croissant:EvaluationTask",
                    "@id": "https://huggingface.co/datasets/c-i-ber/Nova_evaluation_gemini_2_5_flash#image_captioning",
                    "schema:isBasedOn": { "@id": "https://huggingface.co/datasets/c-i-ber/Nova#image_captioning" },
                    "schema:name": "Task 2: Image Captioning",
                    "croissant:evaluatedSolution": { "@id": "https://huggingface.co/datasets/c-i-ber/Nova_solution_gemini_2_5_flash#image_captioning_sol" },
                    "croissant:evaluationResults": [
                        { "@type": "croissant:EvaluationResult", "croissant:metric": "Clinical F1", "croissant:value": str(eval_metrics_dict.get("Clinical F1", 0.0)) },
                        { "@type": "croissant:EvaluationResult", "croissant:metric": "Modality F1", "croissant:value": str(eval_metrics_dict.get("Modality F1", 0.0)) },
                        { "@type": "croissant:EvaluationResult", "croissant:metric": "BLEU-4", "croissant:value": str(eval_metrics_dict.get("BLEU-4", 0.0)) },
                        { "@type": "croissant:EvaluationResult", "croissant:metric": "METEOR", "croissant:value": str(eval_metrics_dict.get("METEOR", 0.0)) },
                        { "@type": "croissant:EvaluationResult", "croissant:metric": "Binary F1", "croissant:value": str(eval_metrics_dict.get("Binary F1", 0.0)) }
                    ]
                }
            },
            {
                "@type": "croissant:TaskSolution",
                "@id": "https://huggingface.co/datasets/c-i-ber/Nova_solution_gemini_2_5_flash#diagnostic_reasoning_sol",
                "schema:isBasedOn": { "@id": "https://huggingface.co/datasets/c-i-ber/Nova#diagnostic_reasoning" },
                "croissant:execution": { "@id": "https://huggingface.co/datasets/c-i-ber/Nova_solution_gemini_2_5_flash#execution" },
                "croissant:implementation": { "@id": "https://huggingface.co/datasets/c-i-ber/Nova_solution_gemini_2_5_flash#implementation" },
                "croissant:output": {
                    "@type": "schema:Dataset",
                    "@id": "urn:uuid:nova-solution-gemini_2_5_flash-diagnostic_reasoning-output"
                },
                "croissant:evaluation": {
                    "@type": "croissant:EvaluationTask",
                    "@id": "https://huggingface.co/datasets/c-i-ber/Nova_evaluation_gemini_2_5_flash#diagnostic_reasoning",
                    "schema:isBasedOn": { "@id": "https://huggingface.co/datasets/c-i-ber/Nova#diagnostic_reasoning" },
                    "schema:name": "Task 3: Multimodal Diagnostic Reasoning",
                    "croissant:evaluatedSolution": { "@id": "https://huggingface.co/datasets/c-i-ber/Nova_solution_gemini_2_5_flash#diagnostic_reasoning_sol" },
                    "croissant:evaluationResults": [
                        { "@type": "croissant:EvaluationResult", "croissant:metric": "Top-1", "croissant:value": str(eval_metrics_dict.get("Top-1", 0.0)) },
                        { "@type": "croissant:EvaluationResult", "croissant:metric": "Top-5", "croissant:value": str(eval_metrics_dict.get("Top-5", 0.0)) },
                        { "@type": "croissant:EvaluationResult", "croissant:metric": "Coverage", "croissant:value": str(eval_metrics_dict.get("Coverage", 0.0)) },
                        { "@type": "croissant:EvaluationResult", "croissant:metric": "Entropy", "croissant:value": str(eval_metrics_dict.get("Entropy", 0.0)) }
                    ]
                }
            }
        ]
    }


def save_solution_file(solution_dict, destination_path):
    """Saves the dictionary as a JSON-LD solution file."""
    with open(destination_path, 'w') as f:
        json.dump(solution_dict, f, indent=2)
