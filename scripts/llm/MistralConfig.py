import torch


class MistralConfig:
    def __init__(self, homedir: str):
        self.paths = {
            "home_dir": homedir,
            "models_dir": f"{homedir}/models",
            "base_model": "mistralai/Mistral-7B-Instruct-v0.1",
            "cache_dir": f"{homedir}/models/cache/",
            "training_dir": f"{homedir}/models/training/"
        }
        self.quantization = {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.bfloat16,
            "bnb_4bit_use_double_quant": False
        }
        self.lora = {
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj", "lm_head"]
        }
        self.training = {
            "output_dir": f"{homedir}/models/training/",
            "num_train_epochs": 1,
            "per_device_train_batch_size": 8,
            "gradient_accumulation_steps": 1,
            "optim": "paged_adamw_8bit",
            "save_steps": 5000,
            "logging_steps": 50,  # round multiple of gradient acc step
            "do_eval": True,
            "evaluation_strategy": "steps",
            "eval_steps": 1000,
            "learning_rate": 2e-5,
            "weight_decay": 0.001,
            "fp16": False,
            "bf16": False,
            "max_grad_norm": 0.3,
            "max_steps": -1,
            "warmup_ratio": 0.3,
            "group_by_length": True,
            "lr_scheduler_type": "constant",
            "report_to": ["wandb"]
        }
        self.trainer = {
            "max_seq_length": None,
            "packing": False
        }
        self.generation = {
            "do_sample": True,
            "num_beams": 5,
            "max_new_tokens": 50,
            "repetition_penalty": 1.3,
            # "exponential_decay_length_penalty": (),
            "temperature": 0.4,
            "length_penalty": -2.0
        }
