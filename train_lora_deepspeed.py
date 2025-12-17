"""
使用 DeepSpeed 进行 LoRA 微调的训练脚本
支持单机单卡/多卡以及多机多卡训练
"""

import os
import argparse
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import torch
import bitsandbytes as bnb

def parse_args():
    parser = argparse.ArgumentParser(description="LoRA Training with DeepSpeed")
    
    # 模型和数据配置
    parser.add_argument(
        "--model_path",
        type=str,
        default="/mnt/workspace/. cache/modelscope/models/qwen/Qwen2-1.5B-Instruct",
        help="本地模型路径"
    )
    parser.add_argument(
        "--data_prefix",
        type=str,
        default="/mnt/workspace/modelscope/data/",
        help="数据集文件路径前缀"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./qwen2-1.5b-en2zh-qlora-deepspeed",
        help="输出目录"
    )
    
    # 训练参数
    parser.add_argument("--num_samples", type=int, default=1000, help="训练样本数量")
    parser.add_argument("--max_seq_length", type=int, default=512, help="最大序列长度")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="每个设备的 batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="学习率")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--logging_steps", type=int, default=10, help="日志记录步数")
    parser.add_argument("--save_steps", type=int, default=100, help="模型保存步数")
    
    # LoRA 参数
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    
    # DeepSpeed 配置
    parser.add_argument(
        "--deepspeed",
        type=str,
        default=None,
        help="DeepSpeed 配置文件路径"
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    
    # 量化配置
    parser.add_argument("--use_4bit", action="store_true", help="是否使用 4-bit 量化")
    
    return parser.parse_args()

def load_data(en_file, zh_file, num_samples):
    """加载训练数据"""
    print(f"🔄 Loading {num_samples} samples from dataset...")
    en_lines, zh_lines = [], []
    
    with open(en_file, encoding="utf-8") as f_en, open(zh_file, encoding="utf-8") as f_zh:
        for i, (en_line, zh_line) in enumerate(zip(f_en, f_zh)):
            if i >= num_samples:
                break
            en_line = en_line.strip()
            zh_line = zh_line.strip()
            if en_line and zh_line:
                en_lines.append(en_line)
                zh_lines.append(zh_line)
    
    print(f"✅ Loaded {len(en_lines)} sentence pairs.")
    return Dataset.from_dict({"en":  en_lines, "zh": zh_lines})

def find_all_linear_names(model):
    """查找模型中所有的线性层（用于 LoRA）"""
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return list(lora_module_names)

def main():
    args = parse_args()
    
    # 设置环境变量
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    print("=" * 60)
    print("🚀 LoRA Training with DeepSpeed")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Output:  {args.output_dir}")
    print(f"Samples: {args.num_samples}")
    print(f"DeepSpeed Config: {args.deepspeed}")
    print(f"Use 4-bit: {args.use_4bit}")
    print("=" * 60)
    
    # ----------------------------
    # 1. 加载 Tokenizer
    # ----------------------------
    print("\n🔄 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=False
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # ----------------------------
    # 2. 加载模型
    # ----------------------------
    print("🔄 Loading model...")
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto" if args.local_rank == -1 else {"":  args.local_rank},
    }
    
    if args.use_4bit:
        print("   Using 4-bit quantization...")
        model_kwargs. update({
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_compute_dtype": torch.bfloat16,
        })
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16
    
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    
    if args.use_4bit:
        model. gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)
    
    # ----------------------------
    # 3. 配置 LoRA
    # ----------------------------
    print("🔄 Configuring LoRA...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=args. lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # ----------------------------
    # 4. 加载数据
    # ----------------------------
    dataset = load_data(
        args.data_prefix + "WikiMatrix.en-zh.en",
        args.data_prefix + "WikiMatrix.en-zh.zh",
        args.num_samples
    )
    
    # ----------------------------
    # 5. 数据格式化函数
    # ----------------------------
    def formatting_func(example):
        if not isinstance(example, dict):
            raise TypeError(f"Expected dict, got {type(example)}")
        if "en" not in example or "zh" not in example:
            raise KeyError(f"Missing 'en' or 'zh' in example")
        
        prompt = (
            "You are a professional translator.\n"
            "<|im_start|>user\n"
            f"Translate the following English text to Chinese:\n{example['en']}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            f"{example['zh']}<|im_end|>"
        )
        inputs = tokenizer(prompt, truncation=True, max_length=args.max_seq_length, add_special_tokens=True)
        return tokenizer.decode(inputs["input_ids"], skip_special_tokens=False)
    
    # ----------------------------
    # 6. 训练参数
    # ----------------------------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        bf16=True,
        optim="paged_adamw_8bit" if args.use_4bit else "adamw_torch",
        lr_scheduler_type="constant",
        report_to="tensorboard",
        deepspeed=args.deepspeed,
        local_rank=args.local_rank,
        ddp_find_unused_parameters=False,
        weight_decay=0.01,
    )
    
    # ----------------------------
    # 7. 初始化 Trainer
    # ----------------------------
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        formatting_func=formatting_func,
        processing_class=tokenizer,
    )
    
    # ----------------------------
    # 8. 开始训练
    # ----------------------------
    print("\n🚀 Starting training...")
    trainer.train()
    
    # ----------------------------
    # 9. 保存模型
    # ----------------------------
    print("💾 Saving model...")
    trainer.save_model()
    print(f"✅ Model saved to {args. output_dir}")
    print("✅ Training completed!")

if __name__ == "__main__": 
    main()
