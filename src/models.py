import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Tuple, Dict, List, Any

# Load tokenizer and causal LM on multiple GPUs
def load_llm(model_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map='auto',
        trust_remote_code=True,
    )
    model.config.pad_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.eos_token_id
    model.eval()

    return model, tokenizer

# Generate model's outputs
def generate_outputs(
    model: AutoModelForCausalLM, 
    tokenizer: AutoTokenizer, 
    prompts: List[List[Dict[str, str]]],
    generation_kwargs: Dict[str, Any],
    batch_size: int = 4,
    return_hidden_states: bool = False,
) -> Tuple[List[str], List[str]]:
    generations = []

    # Detect if using Qwen
    is_qwen = "qwen" in tokenizer.name_or_path.lower()
    if is_qwen:
        for p in prompts:
            assert p[0]['role'] == 'system'
            p[0]['content'] = 'You are Qwen, created by Alibaba Cloud. ' + p[0]['content']

    # Iterate through prompts in batches
    last_token_hidden_states = []
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i+batch_size]

        # Apply chat template
        batch_texts = [
            tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True
            ) for prompt in batch_prompts
        ]

        # Tokenize texts
        model_inputs = tokenizer(batch_texts, return_tensors='pt', padding=True)
        
        # Move inputs to model's first device
        input_device = model.get_input_embeddings().weight.device
        model_inputs = {
            k: v.to(input_device) for k, v in model_inputs.items()
        }

        # Generate outputs
        generated_ids = model.generate(**model_inputs, **generation_kwargs)

        # Remove input tokens for each returned sequence.
        input_lengths = model_inputs["input_ids"].shape[1]
        generated_ids = generated_ids[:, input_lengths:]

        # Decode outputs
        batch_generations = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # Append generations
        generations.extend(batch_generations)

        # Get hidden states
        if return_hidden_states:
            with torch.no_grad():
                outputs = model(**model_inputs, output_hidden_states=True)
            last_token_hidden_state = outputs.hidden_states[-1][:,-1,:]
            last_token_hidden_state = last_token_hidden_state.to(torch.float16).cpu().numpy()
            last_token_hidden_states.append(last_token_hidden_state)

    if return_hidden_states:
        last_token_hidden_states = np.concatenate(last_token_hidden_states, axis=0)
        return generations, last_token_hidden_states
    return generations  