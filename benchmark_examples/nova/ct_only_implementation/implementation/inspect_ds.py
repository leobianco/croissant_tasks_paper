from datasets import load_dataset, Image

ds = load_dataset("parquet", data_files="hf://datasets/c-i-ber/Nova/data/nova-v1.parquet", split="train")
ds = ds.cast_column("image", Image())
row = ds[0]
print(f"image_path: {row['image_path']}")


