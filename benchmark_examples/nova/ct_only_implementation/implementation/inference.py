import functools
import json
import os
import time
from google import genai
from google.genai import types
from nova_results.prompts import (
    ABNORMALITY_GROUNDING_PROMPT,
    DIFFERENTIAL_DIAGNOSIS_PROMPT,
    IMAGE_CAPTIONING_PROMPT,
    IMAGE_DESCRIPTION_EVALUATION_PROMPT,
    MEDICAL_DIAGNOSIS_EVALUATION_PROMPT,
)
import PIL.Image

GLOBAL_CONFIG = types.GenerateContentConfig(
    temperature=0.1, max_output_tokens=4096
)


def retry_on_503(func):

  @functools.wraps(func)
  def wrapper(*args, **kwargs):
    retries = 0
    while retries < 5:
      try:
        return func(*args, **kwargs)
      except Exception as e:
        if (
            "503" in str(e)
            or "UNAVAILABLE" in str(e)
            or "high demand" in str(e).lower()
        ):
          print(
              f"Temporary API error encountered in {func.__name__}, retrying in"
              f" 2 seconds... (Attempt {retries+1}/5)"
          )
          time.sleep(2)
          retries += 1
          continue
        else:
          raise e
    print(f"Max retries reached in {func.__name__}.")
    raise Exception(
        f"Max retries reached in {func.__name__} due to temporary API errors."
    )

  return wrapper


def get_genai_client():
  """Initializes the Google Gen AI client using the GEMINI_API_KEY."""
  api_key = os.environ.get("GEMINI_API_KEY")
  if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set.")
  return genai.Client(api_key=api_key)


@retry_on_503
def run_anomaly_localization(client, image):
  """Calls Gemini 2.5 Flash to predict bounding boxes."""

  try:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[image, ABNORMALITY_GROUNDING_PROMPT],
        config=GLOBAL_CONFIG,
    )
    return response.text
  except Exception as e:
    print(f"Error in run_anomaly_localization: {e}")
    return "[]"


@retry_on_503
def run_image_captioning(client, image):
  """Calls Gemini 2.5 Flash to generate medical descriptions."""

  try:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[image, IMAGE_CAPTIONING_PROMPT],
        config=GLOBAL_CONFIG,
    )
    return response.text
  except Exception as e:
    print(f"Error in run_image_captioning: {e}")
    return ""


@retry_on_503
def run_differential_diagnosis(client, clinical_history, image_caption):
  """Calls Gemini 2.5 Flash to predict differential diagnoses."""

  try:
    prompt = DIFFERENTIAL_DIAGNOSIS_PROMPT.format(
        clinical_history=clinical_history, mri_findings=image_caption
    )
    response = client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt, config=GLOBAL_CONFIG
    )
    return response.text
  except Exception as e:
    print(f"Error in run_differential_diagnosis: {e}")
    return "{}"


@retry_on_503
def run_image_description_evaluation(client, gt_input, pred_input):
  """Calls Gemini 2.5 Flash to extract and standardize keywords."""

  try:
    prompt = IMAGE_DESCRIPTION_EVALUATION_PROMPT.format(
        gt_input=gt_input, pred_input=pred_input
    )
    response = client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt, config=GLOBAL_CONFIG
    )
    return response.text
  except Exception as e:
    print(f"Error in run_image_description_evaluation: {e}")
    return "{}"


@retry_on_503
def run_medical_diagnosis_evaluation(client, gt_diagnosis, pred_diagnosis):
  """Calls Gemini 2.5 Flash to evaluate diagnoses semantically."""

  try:
    prompt = MEDICAL_DIAGNOSIS_EVALUATION_PROMPT.format(
        gt_diagnosis=gt_diagnosis, pred_diagnosis=pred_diagnosis
    )
    response = client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt, config=GLOBAL_CONFIG
    )
    return response.text
  except Exception as e:
    print(f"Error in run_medical_diagnosis_evaluation: {e}")
    return "{}"
