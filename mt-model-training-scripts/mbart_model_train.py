import os
import torch 

os.environ['TRANSFORMERS_CACHE'] = '/scratch/prashantk/cache'
os.environ['HF_DATASETS_CACHE']="/scratch/prashantk/cache"

from huggingface_hub import login
login(token="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")

# loading mBART model
model_checkpoint = "facebook/mbart-large-cc25"

# importing metric and dataset
from datasets import load_dataset, load_metric

raw_datasets = load_dataset("kapilrk04/codemix-en_enhi", use_auth_token=True)
metric = load_metric("sacrebleu")

# tokenizer setup
from transformers import AutoTokenizer
    
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenizer.src_lang = "en-XX"
tokenizer.tgt_lang = "hi-IN"

prefix = ""

max_input_length = 128
max_target_length = 128
source_lang = "en"
target_lang = "en-hi"

def preprocess_function(examples):
    inputs = [prefix + ex[source_lang] for ex in examples["translation"]]
    targets = [ex[target_lang] for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# loading model
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# creating train-test split and preprocessing
raw_datasets = raw_datasets["train"].train_test_split(test_size=0.2)
test_datasets = raw_datasets["test"].train_test_split(test_size=0.004)

tokenized_raw_datasets = raw_datasets.map(preprocess_function, batched=True)
tokenized_test_datasets = test_datasets.map(preprocess_function, batched=True)
#postprocessing and metric calc functions
import numpy as np

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

#training
training_args = Seq2SeqTrainingArguments(
    output_dir="/scratch/prashantk/mbart_based_en_enhi_mt_model",
    evaluation_strategy="no",
    eval_steps = 10,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    save_total_limit=10,
    predict_with_generate=False,
    fp16=False,
    push_to_hub=False,
    report_to="wandb",
    run_name="mbart_en_enhi_mt_model"
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_raw_datasets["train"],
    eval_dataset=tokenized_test_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
