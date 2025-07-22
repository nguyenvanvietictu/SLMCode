# Set up GPU before importing everything else
import os, time
start_time = time.time()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import gc
from random import randrange, random
import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
    pipeline,
    TrainerCallback,
    Trainer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)
from transformers.integrations.tpu import tpu_spmd_dataloader
from trl import SFTTrainer
from huggingface_hub import login
import numpy as np
import warnings, wandb
import evaluate

end_load_time = time.time()

rouge = evaluate.load("rouge")

wandb.login(key="2007c9552166a53af0c8ad3d12cea5c74b281f69") #wandb Viet Nguyen Van

warnings.filterwarnings("ignore")

def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()

clear_memory()

os.environ["WANDB_NOTEBOOK_NAME "] = "SMALL_LANGUAGE_MODEL"
os.environ["WANDB_PROJECT"] = "SMALL_LANGUAGE_MODEL"
os.environ["HF_HUB_TOKEN"] = "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"

login(token=os.getenv("HF_HUB_TOKEN"))
local_model_dir = "./SMALL_LANGUAGE_MODEL"

models = [
    # {"name": "meta-llama/Llama-3.2-1B-Instruct", "has_system":True, "eval_batch_size": 1, "train_batch_size": 1}, #1
    {"name": "meta-llama/Llama-3.2-3B-Instruct", "has_system":True, "eval_batch_size": 1, "train_batch_size": 1}, #2
    # {"name": "microsoft/Phi-3.5-mini-instruct", "has_system":True, "eval_batch_size": 1, "train_batch_size": 1},  #3
    # {"name": "microsoft/Phi-3.5-MoE-instruct", "has_system":True, "eval_batch_size": 1, "train_batch_size": 1},  #4
    # {"name": "nvidia/Hymba-1.5B-Instruct", "has_system":True, "eval_batch_size": 1, "train_batch_size": 1},       #25
    # {"name": "Qwen/Qwen2.5-Coder-3B-Instruct", "has_system":True},
    # {"name": "Qwen/Qwen2.5-Coder-1.5B-Instruct", "has_system":True},
    # {"name": "google/gemma-2-2b-it", "has_system":False},
]

for model in models:
    # 'model_id' and 'model_name' are the identifiers for the pre-trained model from Hugging Face hub that you want to fine-tune.
    model_id = model["name"]
    model_name = model["name"]
    new_model = "SMALL_LANGUAGE_MODEL"
    hf_model_repo="nguyenvanviet/"+new_model
    dataset_name = "sahil2801/CodeAlpaca-20k"
    local_model_dir = "./SMALL_LANGUAGE_MODEL/" + model_name    

    # Load Model on GPU
    max_seq_length = 1024
    device_map = "auto"

    # Bits and Bytes configuration for the model
    use_4bit = True
    bnb_4bit_compute_dtype = "bfloat16"
    bnb_4bit_quant_type = "nf4"
    use_double_quant = True

    # LoRA configuration for the model
    lora_r = 8
    lora_alpha = 8
    lora_dropout = 0.05
    target_modules= ['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"]
    set_seed(1234)

    isSetupParams = True
    if isSetupParams:
        dataset = load_dataset("sahil2801/CodeAlpaca-20k", split="train[:10000]")
        patience = 5
        train_batch_size = 4
        eval_batch_size = 4
        evalsteps = 100
        logsteps = 100
        num_train_epochs = 10
    else:
        dataset = load_dataset("sahil2801/CodeAlpaca-20k", split="train[:18000]")
        patience = 10
        train_batch_size = 2
        eval_batch_size = 2
        evalsteps = 100
        logsteps = 100
        num_train_epochs = 10

    tokenizer_id = model_id
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    def create_message_column(examples):
        out = examples['output'].replace('{', '\\{').replace('}', '\\}')
        ins = examples['instruction'].replace('{', '\\{').replace('}', '\\}')
        inp = examples['input'].replace('{', '\\{').replace('}', '\\}')
        if "gemma" in model_name:
            messages = "<start_of_turn>user\n" + ins + "\n\n" + inp + "<end_of_turn>\n<start_of_turn>model\n" + out + "\n<end_of_turn>"
        else:
            messages = []
            if ins:
                user = {
                    "content": ins,
                    "role": "user"
                }
                messages.append(user)
            if inp:
                user = {
                    "content": inp,
                    "role": "user"
                }
                messages.append(user)
            assistant = {
                "content": out,
                "role": "assistant"
            }
            messages.append(assistant)
        
        return {"messages": messages,}

    def format_dataset_chatml(row):
        
        if "gemma" in model_name:
            text = row["messages"]
        else:
            text = tokenizer.apply_chat_template(row["messages"], add_generation_prompt=False, tokenize=False)
        tokenized_inputs = tokenizer(text, padding='max_length', truncation=True, max_length=max_seq_length)
        labels = tokenizer(row["output"], padding='max_length', truncation=True, max_length=max_seq_length)["input_ids"]
        # drop instruction, input, output, messages columns
        del row["instruction"]
        del row["input"]
        del row["output"]
        del row["messages"]

        return {
            "text": text,
            "input_ids": tokenized_inputs["input_ids"],
            "attention_mask": tokenized_inputs["attention_mask"],
            "labels": labels,
        }

    dataset_chatml = dataset.map(create_message_column)
    dataset_chatml = dataset_chatml.map(format_dataset_chatml)
    dataset_chatml = dataset_chatml.train_test_split(test_size=0.1, seed=1234)

    # print(dataset_chatml['train'][0])

    if torch.cuda.is_bf16_supported():
        compute_dtype = torch.bfloat16
        attn_implementation = 'flash_attention_2'
    else:
        compute_dtype = torch.float16
        attn_implementation = 'sdpa'

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_double_quant,
    )

    model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=compute_dtype, trust_remote_code=True, quantization_config=bnb_config, device_map=device_map,
            attn_implementation=attn_implementation
    )

    model = prepare_model_for_kbit_training(model)

    run_name = model_name

    args = TrainingArguments(
        output_dir=local_model_dir,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=evalsteps,
        # do_eval=False,
        optim="adamw_torch",
        per_device_train_batch_size=train_batch_size,
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=eval_batch_size,
        log_level="debug",
        learning_rate=1e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        eval_steps=evalsteps,
        logging_steps=logsteps,
        num_train_epochs=num_train_epochs,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        report_to="wandb",
        seed=342,
        # load_best_model_at_end=True,
        # metric_for_best_model="total_rouge",
        # greater_is_better=True,
        run_name=run_name,
        prediction_loss_only=False,
        label_names=["labels"],
        disable_tqdm=False,
    )

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
    )

    data_collator = DataCollatorWithPadding(tokenizer, padding=True)

    def compute_metrics(eval_pred):
        #fake total_rouge
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        predictions = [[32000 if x==8689 else x for x in pred] for pred in predictions]
        predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        rouge_scores = rouge.compute(predictions=predictions, references=labels)
        total_rouge = sum(rouge_scores.values())/len(rouge_scores)
        rouge_scores["eval_total_rouge"] = total_rouge
        return rouge_scores

    packing = True
    if "gemma" in model_name:
        packing = True

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset_chatml['train'],
        eval_dataset=dataset_chatml['test'],
        peft_config=peft_config,
        #dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        #compute_metrics=compute_metrics,   # Tat bo eval
        packing=packing,
        # callbacks=[early_stopping],
    )

    trainer.train()
    # trainer.save_model(local_model_dir)

    # trainer.push_to_hub(hf_model_repo)

    # tokenizer.save_pretrained(local_model_dir)
    # model.save_pretrained(local_model_dir)

    #Show current memory stats
    torch.cuda.synchronize()
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    del model
    del tokenizer
    del trainer
    general_end_time = time.time()
    clear_memory()
    torch.cuda.synchronize()

    # Show total time
    print(f"Total time: {general_end_time - start_time} seconds")
    print(f"Load time: {end_load_time - start_time} seconds")