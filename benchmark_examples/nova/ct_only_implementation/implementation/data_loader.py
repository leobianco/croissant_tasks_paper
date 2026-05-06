import os
import requests
from PIL import Image
from io import BytesIO
from datasets import load_dataset

def load_nova_dataset():
    """Loads the NOVA dataset from Hugging Face."""
    try:
        ds = load_dataset("parquet", data_files="hf://datasets/c-i-ber/Nova/data/nova-v1.parquet", split="train")
        return ds
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise e

def download_image(image_path):
    """Downloads the brain MRI image from Hugging Face."""
    url = f"https://huggingface.co/datasets/c-i-ber/Nova/resolve/main/{image_path}"
    print(f"Fetching image from {url}")
    try:
        response = requests.get(url)
        print(f"Image fetch status code: {response.status_code}")
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
        else:
            print(f"Failed to fetch image {url}, status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error downloading image {url}: {e}")
        return None



