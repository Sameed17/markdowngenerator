import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel
from evaluate import load

# =========================
# CONFIG
# =========================
MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
CHECKPOINT_PATH = "./qlora-vlm"
JSONL_PATH = "0508_clean.jsonl"

NUM_SAMPLES = 3
IMAGE_SIZE = 512
MAX_NEW_TOKENS = 1024

# =========================
# LOAD PROCESSOR
# =========================
processor = AutoProcessor.from_pretrained(MODEL_NAME)

# =========================
# LOAD 4-BIT CONFIG
# =========================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# =========================
# LOAD ZERO-SHOT MODEL
# =========================
print("Loading zero-shot model...")
base_model_zero = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)
base_model_zero.eval()

# =========================
# LOAD FINE-TUNED MODEL
# =========================
print("Loading fine-tuned model...")
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
dataset = load_dataset("json", data_files=JSONL_PATH)["train"]

# =========================
# HELPER: GENERATION
# =========================
def build_inputs(image):
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
    )

    return inputs

def generate(model_obj, image_path):
    image = Image.open(image_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    inputs = build_inputs(image).to(model_obj.device)

    with torch.no_grad():
        output = model_obj.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False
        )

    return processor.decode(output[0], skip_special_tokens=True)

# =========================
# OPTIONAL: METRIC
# =========================
def compute_rouge(pred, gt):
    rouge = load("rouge")
    score = rouge.compute(predictions=[pred], references=[gt])
    return score


# =========================
# VISUALIZATION
# =========================
def visualize(sample, title="Sample", save=False):
    image_path = sample["image"]
    gt = sample["markdown"]

    pred_zero = generate(base_model_zero, image_path)
    pred_ft = generate(model, image_path)

    image = Image.open(image_path).convert("RGB")

    # show image
    plt.figure(figsize=(8, 5))
    plt.imshow(image)
    plt.axis("off")
    plt.title(title)
    if save:
        plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.show()

    print("\n" + "="*100)
    print("📌 GROUND TRUTH:\n")
    print(gt[:1500])

    print("\n" + "-"*100)
    print("🤖 ZERO-SHOT OUTPUT:\n")
    print(pred_zero[:1500])

    print("\n" + "-"*100)
    print("🚀 FINE-TUNED OUTPUT:\n")
    print(pred_ft[:1500])

    # optional metric
    rouge_scores = compute_rouge(pred_ft, gt)
    print("\n📊 ROUGE (Fine-tuned):", rouge_scores)

    print("="*100)

# =========================
# RUN: TRAIN SAMPLES
# =========================
print("\n===== TRAIN SAMPLES =====\n")
for i in range(NUM_SAMPLES):
    visualize(dataset[i], title=f"Train Sample {i+1}")