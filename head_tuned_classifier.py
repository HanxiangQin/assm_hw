import torch
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import pipeline
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoModelForCausalLM
from tqdm import tqdm
import os
from peft import LoraConfig, get_peft_model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_id = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)


ds = load_dataset("wics/strategy-qa", revision="refs/convert/parquet", split="test")

# Split dataset into train (70%), dev (15%), and test (15%)
train_test_split = ds.train_test_split(test_size=0.3, seed=42)
train_ds = train_test_split['train']
temp_ds = train_test_split['test']
dev_test_split = temp_ds.train_test_split(test_size=0.5, seed=42)
dev_ds = dev_test_split['train']
test_ds = dev_test_split['test']

print(f"Dataset splits:")
print(f"  Train: {len(train_ds)} examples")
print(f"  Dev: {len(dev_ds)} examples")
print(f"  Test: {len(test_ds)} examples")

def tokenize_function(examples):
    text = [
        term + tokenizer.sep_token + description + tokenizer.sep_token + question + tokenizer.sep_token + f"Facts: {facts}"
        for term, description, question, facts in zip(examples['term'], examples['description'], examples['question'], examples['facts'])
    ]
    tokenized = tokenizer(text, padding=True, truncation=True)
    # Convert boolean labels to integers (True -> 1, False -> 0)
    tokenized['labels'] = [int(label) for label in examples['answer']]
    return tokenized
# Tokenize all splits
train_tokenized = train_ds.map(tokenize_function, batched=True)
dev_tokenized = dev_ds.map(tokenize_function, batched=True)
test_tokenized = test_ds.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

use_lora = True
model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)
for name, param in model.named_parameters():
    print(name, param.shape)
print("--------------------------------")
print(model)
if use_lora:
    print("Using LoRA for training.")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["classifier"],
        lora_dropout=0.05,
        bias="none",
        # task_type="SEQ_CLS"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
else:
    print("Using head-tuning for training.")
    # Freeze base model parameters first
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if 'classifier' in name:
            param.requires_grad = True

model.to(device)

accuracy_metrics = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": accuracy_metrics.compute(predictions=predictions, references=labels)}


training_args = TrainingArguments(
    output_dir="output/modernbert-base-classifier-finetuned",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    save_total_limit=3,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=20,
    learning_rate=5e-3,
    weight_decay=0.001,
    logging_dir="logs",
    logging_steps=10,
    logging_first_step=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=dev_tokenized,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate(test_tokenized)
trainer.save_model("output/modernbert-base-classifier-finetuned")
