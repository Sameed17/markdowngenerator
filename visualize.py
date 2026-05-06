import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel

# =========================
# CONFIG
# =========================
MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
CHECKPOINT_PATH = "./qlora-vlm"   # your saved model
JSONL_PATH = "0508_clean.jsonl"

NUM_SAMPLES = 3   # show 3 samples
IMAGE_SIZE = 512

# =========================
# LOAD MODEL (IMPORTANT)
# =========================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

processor = AutoProcessor.from_pretrained(MODEL_NAME)

base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH)
model.eval()

# =========================
# LOAD DATASET
# =========================
dataset = load_dataset("json", data_files=JSONL_PATH)

# =========================
# GENERATION FUNCTION
# =========================
def generate_markdown(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

    prompt = "Convert this document image into structured Markdown."

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=text,
        images=image,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False
        )

    result = processor.decode(output[0], skip_special_tokens=True)
    return result

# =========================
# VISUALIZATION FUNCTION
# =========================
def visualize_sample(sample, title="Sample"):
    image_path = sample["image"]
    gt_markdown = sample["markdown"]

    pred_markdown = generate_markdown(image_path)

    image = Image.open(image_path).convert("RGB")

    plt.figure(figsize=(10, 6))
    plt.imshow(image)
    plt.axis("off")
    plt.title(title)
    plt.show()

    print("\n" + "="*80)
    print("📌 GROUND TRUTH MARKDOWN:\n")
    print(gt_markdown[:2000])

    print("\n" + "-"*80)
    print("🤖 GENERATED MARKDOWN:\n")
    print(pred_markdown[:2000])
    print("="*80)

# =========================
# RUN VISUALIZATION
# =========================
print("\n===== TRAIN SAMPLES =====\n")
for i in range(NUM_SAMPLES):
    visualize_sample(dataset[i], title=f"Train Sample {i+1}")