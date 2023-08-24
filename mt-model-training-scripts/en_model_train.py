import os
import torch 

os.environ['TRANSFORMERS_CACHE'] = '/scratch/prashantk/cache'
os.environ['HF_DATASETS_CACHE']="/scratch/prashantk/cache"

from huggingface_hub import login

from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

import evaluate

import numpy as np

from datasets import load_dataset

source_lang = "en"
target_lang = "en-hi"
prefix = "Translate English to Hinglish: "

def preprocess_function(examples):
    inputs = [prefix + example[source_lang] for example in examples["translation"]]
    targets = [example[target_lang] for example in examples["translation"]]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
    return model_inputs

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

login(token="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")

checkpoint = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)

metric = evaluate.load("sacrebleu")

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
en_dataset = load_dataset("kapilrk04/codemix-en_enhi", use_auth_token=True)
en_dataset = en_dataset["train"].train_test_split(test_size=0.2)

tokenized_en_dataset = en_dataset.map(preprocess_function, batched=True)

training_args = Seq2SeqTrainingArguments(
    output_dir="/scratch/prashantk/mt5_based_en_enhi_mt_model",
    evaluation_strategy="steps",
    eval_steps = 5000,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    save_total_limit=10,
    predict_with_generate=True,
    fp16=False,
    push_to_hub=True,
    report_to="wandb",
    run_name="en_enhi_mt_model"
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_en_dataset["train"],
    eval_dataset=tokenized_en_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

