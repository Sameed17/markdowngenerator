import os
import torch
from PIL import Image
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model

# =========================
# CONFIG
# =========================

JSONL_PATH = "0508_clean.jsonl"
MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
OUTPUT_DIR = "./qlora-vlm"

BATCH_SIZE = 1
GRAD_ACCUM = 4
LR = 2e-4
EPOCHS = 2
MAX_LENGTH = 1024
IMAGE_SIZE = 512
FRAC = 0.01

# =========================
# LOAD MODEL (4-bit)
# =========================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

processor = AutoProcessor.from_pretrained(MODEL_NAME)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

# =========================
# APPLY LORA
# =========================
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# =========================
# DATASET
# =========================
dataset = load_dataset(
    "json",
    data_files=JSONL_PATH
)

dataset = dataset["train"].train_test_split(test_size=0.2)

def preprocess(example):
    # Fix image path
    # image = Image.open(example["image"]).convert("RGB")
    # Clip markdown (important)
    prompt = "Convert this document image into structured Markdown."

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        },
        {"role": "assistant", "content": example["markdown"]}
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    return {
        "text": text,
        "image_path": example["image"]
    }

    # inputs = processor(
    #     text=text,
    #     images=image,
    #     return_tensors="pt",
    #     padding="max_length",
    #     truncation=True,
    #     max_length=MAX_LENGTH
    # )

    # inputs["labels"] = inputs["input_ids"].clone()
    # return {k: v.squeeze(0) for k, v in inputs.items()}

# =========================
# APPLY PREPROCESSING
# =========================

train_ds = dataset["train"].select(range(int(FRAC * len(dataset["train"])))).map(
    preprocess,
    remove_columns=dataset["train"].column_names
)
val_ds = dataset["test"].select(range(int(FRAC * len(dataset["test"])))).map(
    preprocess,
    remove_columns=dataset["test"].column_names
)

# =========================
# COLLATE FUNCTION (KEY FIX)
# =========================
def collate_fn(batch):
    images = []
    texts = []
    for x in batch:
        image = Image.open(x["image_path"]).convert("RGB")
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))  # 🔥 reduce memory
        images.append(image)
        texts.append(x["text"])
    inputs = processor(
        text=texts,
        images=images,
        padding=True,  # 🔥 dynamic padding (IMPORTANT)
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    labels = inputs["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    inputs["labels"] = labels
    return inputs
# =========================
# TRAINING
# =========================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    fp16=True,
    report_to="none",
    remove_unused_columns=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collate_fn
)

trainer.train()

# =========================
# SAVE MODEL
# =========================
model.save_pretrained(OUTPUT_DIR)

# =========================
# INFERENCE (TEST)
# =========================
def generate_markdown(image_path):
    image = Image.open(image_path).convert("RGB")

    prompt = "Convert this document image into structured Markdown."

    messages = [
        {"role": "user", "content": prompt}
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
        output = model.generate(**inputs, max_new_tokens=512)

    result = processor.decode(output[0], skip_special_tokens=True)
    return result

# Example usage
# print(generate_markdown("sample.png"))