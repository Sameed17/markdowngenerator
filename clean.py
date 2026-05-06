import json
import os
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

INPUT_JSONL = "0508.jsonl"
OUTPUT_JSONL = "0508_clean.jsonl"

def is_valid_image(path):
    if not os.path.exists(path):
        return False

    try:
        with Image.open(path) as img:
            img.verify()  # 🔥 checks corruption without loading full image
        return True
    except (UnidentifiedImageError, OSError):
        return False


def clean_jsonl():
    total = 0
    kept = 0
    removed = 0

    with open(INPUT_JSONL, "r", encoding="utf-8") as infile, \
         open(OUTPUT_JSONL, "w", encoding="utf-8") as outfile:

        for line in tqdm(infile, desc="Cleaning dataset"):
            total += 1

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                removed += 1
                continue

            image_path = data.get("image", "")
            markdown = data.get("markdown", "")

            # 🔴 Check 1: image validity
            if not is_valid_image(image_path):
                removed += 1
                continue

            # 🔴 Check 2: empty or tiny markdown
            if not markdown or len(markdown.strip()) < 10:
                removed += 1
                continue

            # 🔴 Optional: cut extremely large samples (prevents future crashes)
            if len(markdown) > 10000:
                data["markdown"] = markdown[:10000]

            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
            kept += 1

    print("\n===== CLEANING SUMMARY =====")
    print(f"Total rows:   {total}")
    print(f"Kept rows:    {kept}")
    print(f"Removed rows: {removed}")
    print(f"Saved to:     {OUTPUT_JSONL}")


if __name__ == "__main__":
    clean_jsonl()