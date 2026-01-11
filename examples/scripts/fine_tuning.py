from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from datasets import load_dataset
import argparse

EOS_TOKEN = None

def formatting_prompts_func(examples):
	instructions = examples["instruction"]
	inputs       = examples["input"]
	outputs      = examples["output"]
	texts = []
	
	alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

		### Instruction:
		{}

		### Input:
		{}

		### Response:
		{}"""

	for instruction, input, output in zip(instructions, inputs, outputs):
		# Must add EOS_TOKEN, otherwise your generation will go on forever!
		text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
		texts.append(text)
	return { "text" : texts, }
	
if __name__ == "__main__":
	
	parser = argparse.ArgumentParser(description="Process PDF files from given folders.")
	parser.add_argument('--jsonl_fp', type=str, help='jsonl file', default='data/preproc/stacked_pdf_trial.jsonl')
	parser.add_argument('--save_dp', type=str, help='Directory where the output files should be saved', default='data/models/lora_model')
	args = parser.parse_args()
	
	max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
	dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
	load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

	# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
	fourbit_models = [
		"unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit",
		"unsloth/Meta-Llama-3.1-8B-bnb-4bit",      # Llama-3.1 15 trillion tokens model 2x faster!
		"unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
		"unsloth/Meta-Llama-3.1-70B-bnb-4bit",
		"unsloth/Meta-Llama-3.1-405B-bnb-4bit",    # We also uploaded 4bit for 405b!
		"unsloth/Mistral-Nemo-Base-2407-bnb-4bit", # New Mistral 12b 2x faster!
		"unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
		"unsloth/mistral-7b-v0.3-bnb-4bit",        # Mistral v3 2x faster!
		"unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
		"unsloth/Phi-3-mini-4k-instruct",          # Phi-3 2x faster!d
		"unsloth/Phi-3-medium-4k-instruct",
		"unsloth/gemma-2-9b-bnb-4bit",
		"unsloth/gemma-2-27b-bnb-4bit",            # Gemma 2x faster!
	] # More models at https://huggingface.co/unsloth

	pretrained_model, tokenizer = FastLanguageModel.from_pretrained(
		model_name = "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
		max_seq_length = max_seq_length,
		dtype = dtype,
		load_in_4bit = load_in_4bit,
		# token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
	)
	
	peft_model = FastLanguageModel.get_peft_model(
		pretrained_model,
		r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
		target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
						"gate_proj", "up_proj", "down_proj",],
		lora_alpha = 16,
		lora_dropout = 0, # Supports any, but = 0 is optimized
		bias = "none",    # Supports any, but = "none" is optimized
		# [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
		use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
		random_state = 3407,
		use_rslora = False,  # We support rank stabilized LoRA
		loftq_config = None, # And LoftQ
	)
	
	# setup dataset
	EOS_TOKEN = tokenizer.eos_token # this is bad but it will do for now
	#jsonl_fp = 'NICHE 1 TRAINING SET_with_instructions4.jsonl'
	dataset = load_dataset('json', data_files=args.jsonl_fp) #load_dataset("150_Training_cleaned.jsonl", split = "train")
	dataset = dataset.map(formatting_prompts_func, batched = True,)

	# setup trainer
	trainer = SFTTrainer(
		model = peft_model,
		tokenizer = tokenizer,
		train_dataset = dataset['train'],
		dataset_text_field = "text",
		max_seq_length = max_seq_length,
		dataset_num_proc = 2,
		packing = False, # Can make training 5x faster for short sequences.
		args = TrainingArguments(
			per_device_train_batch_size = 2,
			gradient_accumulation_steps = 4,
			warmup_steps = 5,
			num_train_epochs = 1, # Set this for 1 full training run.
			max_steps = 10,
			learning_rate = 2e-4,
			fp16 = not is_bfloat16_supported(),
			bf16 = is_bfloat16_supported(),
			logging_steps = 1,
			optim = "adamw_8bit",
			weight_decay = 0.01,
			lr_scheduler_type = "linear",
			seed = 3407,
			output_dir = "outputs",
			save_strategy="epoch",
		),
	)
	
	# training the model
	trainer_stats = trainer.train()
	
	# save model
	peft_model.save_pretrained(args.save_dp) # Local saving
	tokenizer.save_pretrained(args.save_dp)
 
	# offload model from gpiu .... 