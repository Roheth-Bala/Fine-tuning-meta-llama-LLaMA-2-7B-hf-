# train_dolly_lora_smooth_integrated_fixed_v3.py
import os
import json
import torch
import time
import logging
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from huggingface_hub import login

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set Random Seed for Reproducibility
set_seed(42)

# Optional: Login if needed (e.g., for Llama models)
# login(new_session=False)

# --- 1. Load and Preprocess Dataset ---
logger.info("Loading Dolly-15k dataset...")
dataset = load_dataset("databricks/databricks-dolly-15k")

def format_dolly(sample):
    """Formats a Dolly sample into a single text string."""
    instr = sample["instruction"]
    ctx = sample["context"]
    resp = sample["response"]
    # Filter out samples with empty responses
    if not resp or not resp.strip():
        return None
    if ctx and ctx.strip():
        text_output = f"### Instruction:\n{instr}\n\n### Context:\n{ctx}\n\n### Response:\n{resp}"
    else:
        text_output = f"### Instruction:\n{instr}\n\n### Response:\n{resp}"
    # Return a dictionary with the key 'text' which SFTTrainer expects by default
    return {"text": text_output}

# Apply formatting and filter
formatted_dataset = dataset["train"].map(format_dolly).filter(lambda x: x is not None)

# Split the dataset: 80% train, 10% validation, 10% test
train_val = formatted_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_val["train"].train_test_split(test_size=0.125, seed=42) # 0.125 * 0.8 = 0.1 (10% of total)
final_dataset = {
    "train": train_dataset["train"],
    "validation": train_dataset["test"],
    "test": train_val["test"] # Remaining 10% for test
}

logger.info(f"Dataset splits created. Train: {len(final_dataset['train'])}, Val: {len(final_dataset['validation'])}, Test: {len(final_dataset['test'])}")

# --- 2. Model & Tokenizer ---
model_id = "meta-llama/Llama-2-7b-hf"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)

logger.info(f"✓ Model {model_id} loaded in 4-bit (QLoRA)")

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# --- 3. Prepare Model for k-bit Training BEFORE applying PEFT ---
# This ensures quantized layers are handled correctly by PEFT
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
logger.info("Model prepared for k-bit training.")

# --- 4. LoRA Config (Adapted from reference script for potentially smoother updates) ---
peft_config = LoraConfig(
    r=8, # Reduced r from 16 to 8 (like reference script)
    lora_alpha=16, # Reduced alpha from 32 to 16 (like reference script)
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
)

# --- 5. Apply PEFT AFTER preparing for k-bit training ---
model = get_peft_model(model, peft_config)
model.print_trainable_parameters() # Optional: Print trainable params to confirm PEFT worked
logger.info("Model wrapped with PEFT LoRA.")

# --- 6. SFTConfig (Integrating features from reference script) ---
# Removed 'max_length' from here as it's not valid for TrainingArguments
# Adjusted memory-related arguments for smoother convergence and aligned logging
# Set remove_unused_columns=False to avoid issues with test set evaluation
sft_config = SFTConfig(
    output_dir="./results/dolly_lora_llama2_7b_smooth_integrated_fixed_v3",
    num_train_epochs=3, # Increased from 1 to 3 for more gradual learning
    per_device_train_batch_size=4, # Increased from 1 to 4 for smoother gradients
    per_device_eval_batch_size=4, # Increased from 1 to 4
    gradient_accumulation_steps=8, # Adjusted to keep effective batch size similar (4*8 = 32)
    learning_rate=2e-4,
    weight_decay=0.0, # Added from reference script
    warmup_ratio=0.03,
    max_steps=-1, # Use epochs instead
    logging_strategy="steps",
    logging_steps=20, # Increased from 10, aligns better with eval_steps
    eval_strategy="steps", # Use eval_strategy instead of deprecated evaluation_strategy
    eval_steps=50, # Increased from 200, aligns better with logging_steps
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    report_to="none",
    bf16=True, # Align with model's dtype from bnb_config
    # fp16=False, # Removed conflicting fp16
    dataloader_num_workers=2,
    remove_unused_columns=False, # <--- SET TO FALSE to handle external eval dataset correctly
    group_by_length=False, # <--- DISABLED to fix evaluation error from previous script
    gradient_checkpointing=True, # Keep this to save memory (should be consistent with prepare_model_for_kbit_training)
    optim="paged_adamw_32bit", # Consider changing to paged_adamw_8bit for QLoRA consistency if needed
    lr_scheduler_type="cosine",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    dataset_text_field="text", # Specify the text field name
    max_seq_length=1024, # Added from reference script
    packing=False, # Keep packing False as per previous adaptations
    # torch_compile=True, # <--- OPTIONAL: Uncomment if compatible
)

# --- 7. SFTTrainer ---
# Do NOT pass peft_config here, as the model is already PEFT-wrapped
# Removed max_seq_length argument which caused the error in earlier attempts
trainer = SFTTrainer(
    model=model, # Use the PEFT-wrapped model
    args=sft_config,
    train_dataset=final_dataset["train"],
    eval_dataset=final_dataset["validation"],
    # peft_config=peft_config, # <-- REMOVED: Model is already wrapped
    processing_class=tokenizer, # Use processing_class argument (aligns with Colab version and reference script)
    # dataset_text_field="text", # <-- REMOVED from SFTTrainer init, specified in SFTConfig
    # max_seq_length=512, # <-- REMOVED: Specified in SFTConfig
    # packing=False, # <-- REMOVED: Specified in SFTConfig
)

logger.info("\nStarting QLoRA fine-tuning for smoother convergence...")
start_time = time.time()

trainer.train()

end_time = time.time()
training_duration_minutes = (end_time - start_time) / 60
logger.info(f"Training finished in: {training_duration_minutes:.2f} minutes")

# --- 8. Save Final Model ---
final_save_dir = os.path.join(sft_config.output_dir, "final") # Use path from args
trainer.save_model(final_save_dir)
tokenizer.save_pretrained(final_save_dir)
logger.info(f"Final model and tokenizer saved to: {final_save_dir}")

# --- 9. Plot Loss Curves (As requested - MOVED BEFORE TEST LOSS) ---
log_history = trainer.state.log_history
log_save_path = os.path.join(sft_config.output_dir, "training_logs.json") # Use path from args
with open(log_save_path, "w") as f:
    json.dump(log_history, f, indent=2)
logger.info(f"Training logs saved to: {log_save_path}")

# Extract loss values
steps = [entry["step"] for entry in log_history if "loss" in entry]
losses = [entry["loss"] for entry in log_history if "loss" in entry]
eval_steps = [entry["step"] for entry in log_history if "eval_loss" in entry]
eval_losses = [entry["eval_loss"] for entry in log_history if "eval_loss" in entry]

# Plot
plt.figure(figsize=(10, 5))
plt.plot(steps, losses, label="Train Loss", alpha=0.8, marker='o', markersize=2) # Added markers for clarity
if eval_losses:
    plt.plot(eval_steps, eval_losses, label="Eval Loss", alpha=0.8, marker='s', markersize=2) # Added markers for clarity
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training and Validation Loss (Smoothed Convergence)")
plt.legend()
plt.grid(True)
plot_path = os.path.join(sft_config.output_dir, "loss_curve_smooth.png") # Use path from args
plt.savefig(plot_path)
logger.info(f"Loss curve saved to: {plot_path}")

# --- 10. Compute Test Loss ---
logger.info("Computing test loss...")
# Apply the same formatting function to the test set to ensure 'text' column exists
test_dataset_formatted = final_dataset["test"].map(format_dolly).filter(lambda x: x is not None)
test_loss_results = trainer.evaluate(eval_dataset=test_dataset_formatted) # Use the formatted test set
logger.info(f"Test Loss: {test_loss_results['eval_loss']:.4f}")
logger.info(f"Test Perplexity: {np.exp(test_loss_results['eval_loss']):.4f}")

# --- 11. Log Hardware Info ---
# Basic GPU info (more detailed VRAM logging would require nvidia-ml-py)
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
logger.info(f"GPU used: {gpu_name}")
logger.info(f"Estimated VRAM usage (Peak allocated MB): {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")


# Optional: print last few metrics
train_metrics = trainer.state.log_history
if len(train_metrics) > 5:
    logger.info("\nLast 5 log entries:")
    for entry in train_metrics[-5:]:
        logger.info(entry)
else:
    logger.info("\nTraining metrics:", train_metrics)

logger.info("Script completed successfully.")