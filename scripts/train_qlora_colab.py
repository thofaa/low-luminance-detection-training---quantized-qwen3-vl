#!/usr/bin/env python3
"""QLoRA training scaffold for Qwen3-VL-8B-Instruct in Colab."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from peft import LoraConfig
from transformers import TrainingArguments


@dataclass
class TrainConfig:
    base_model: str
    train_jsonl: str
    val_jsonl: str
    output_dir: str
    epochs: int
    learning_rate: float
    batch_size: int


def build_lora_config() -> LoraConfig:
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )


def build_training_args(cfg: TrainConfig) -> TrainingArguments:
    kwargs = {
        "output_dir": cfg.output_dir,
        "per_device_train_batch_size": cfg.batch_size,
        "per_device_eval_batch_size": cfg.batch_size,
        "num_train_epochs": cfg.epochs,
        "learning_rate": cfg.learning_rate,
        "bf16": True,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "report_to": "none",
    }
    try:
        return TrainingArguments(eval_strategy="epoch", **kwargs)
    except TypeError:
        return TrainingArguments(evaluation_strategy="epoch", **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    cfg = TrainConfig(
        base_model=args.base_model,
        train_jsonl=args.train_jsonl,
        val_jsonl=args.val_jsonl,
        output_dir=args.output_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
    )
    lora_cfg = build_lora_config()
    training_args = build_training_args(cfg)
    print("Prepared QLoRA configuration for Colab training:")
    print(f"- base_model={cfg.base_model}")
    print(f"- train={cfg.train_jsonl}")
    print(f"- val={cfg.val_jsonl}")
    print(f"- output_dir={cfg.output_dir}")
    print(f"- lora_targets={lora_cfg.target_modules}")
    eval_strategy = getattr(training_args, "eval_strategy", getattr(training_args, "evaluation_strategy", "unknown"))
    print(f"- eval_strategy={eval_strategy}")
    print("Integrate this script with model/data loading cells in your Colab notebook.")


if __name__ == "__main__":
    main()
