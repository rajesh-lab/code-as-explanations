import datasets
from datasets import Dataset
import numpy as np
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Union, List

ALTERNATIVE_KWARGS = {
    'max_new_tokens': 2048,
    'do_sample': True,
    'temperature': 0.1,
    'top_p': 0.95,
    'top_k': 50,
    'repetition_penalty': 1.2,
}

SCS_PROMPT = """You are given a task prompt, an input passage, and an explanation describing which aspects of the input are important for the task.
Your task is to alter the input passage such that all entities, roles, relationships, and facts mentioned in the explanation are preserved exactly.
The explanation must still fully apply to the edited passage without modification.
Everything else in the passage may be changed freely, including wording, style, or additional context, as long as it does not alter the important elements from the explanation.

First briefly reason about what you can and cannot change about the input passage. Then produce your altered input passage.
The altered input passage should be enclosed exactly between the tags Start Of Passage and End Of Passage."""

def compute_sufficiency(
    base_log_probs: np.ndarray, alt_log_probs: np.ndarray
) -> np.ndarray:
    # Reshape the log probabilities to 2D arrays
    assert base_log_probs.shape == alt_log_probs.shape
    if len(base_log_probs.shape) == 1:
        base_log_probs = base_log_probs.reshape(-1, 1)
        alt_log_probs = alt_log_probs.reshape(-1, 1)

    # Reweighted log probabilities with base loog probs for computation stability
    weights = np.exp(base_log_probs - np.max(base_log_probs, axis=1, keepdims=True))
    base_nll = - (np.sum(weights * base_log_probs, axis=1) / np.sum(weights, axis=1))
    alt_nll = - (np.sum(weights * alt_log_probs, axis=1) / np.sum(weights, axis=1))
    
    # Compute the sufficiency score
    scores = np.array([
        (1 - np.clip((alt - base) / (alt - base + np.log(base_log_probs.shape[1])), 0, 1))
        if (alt + base != 0) else 1
        for base, alt in zip(base_nll, alt_nll)
    ])
    return scores

@torch.no_grad()
def compute_log_probs(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: Union[List[str], List[List[Dict[str, str]]]],
    generations: List[str],
    valid_options: List[str],
    stop_str: str = '\n<FINAL_ANSWER>\n',
) -> np.ndarray:
    device = model.device

    # Convert prompts into list of strings
    if isinstance(prompts[0], list):
        prompts = [
            tokenizer.apply_chat_template(
                p, tokenize=False, add_generation_prompt=True
            ) for p in prompts
        ]

    # Format target options
    if '(A)' in valid_options:
        valid_options = [
            str(option).replace("(", "").replace(")", "")
            for option in valid_options
        ]
    else:
        valid_options = [str(option) for option in valid_options]

    # Convert targets to token IDs, allowing for varying lengths
    target_ids = [
        tokenizer(
            target, add_special_tokens=False
        ).input_ids
        for target in valid_options
    ]

    log_probs_list = []

    # Build modified sequences
    for prompt, generation in zip(prompts, generations):
        if stop_str in generation:
            prefix = generation.split(stop_str)[0] + stop_str
        else:
            prefix = generation + stop_str

        # Track answer span
        prefix_ids = tokenizer(
            prompt + prefix, add_special_tokens=False
        ).input_ids
        start_ind = len(prefix_ids) - 1

        # Compute log probabilities for each target option
        target_log_probs = []
        for target, ids in zip(valid_options, target_ids):
            full_text = prompt + prefix + target

            # Tokenize and forward pass in batch
            model_inputs = tokenizer(
                full_text, return_tensors="pt", add_special_tokens=False,
            ).to(device)

            output = model(**model_inputs, use_cache=False)

            # Only retain logits from the answer region
            logits = output.logits[0, start_ind:-1, :]

            # Preserve original start_k behavior
            predicted_ids = logits.argmax(dim=-1)

            # Find the first target token whose decoded text contains an alphanumeric character
            start_k = 0
            for k, token_id in enumerate(predicted_ids):
                predicted_text = tokenizer.decode([token_id.item()])
                if any(char.isalnum() for char in predicted_text):
                    start_k = k
                    break

            # Select only the target token positions
            end_k = start_k + len(ids)
            if end_k > logits.shape[0]:
                target_log_probs.append(float('-inf'))
                continue
            positions = torch.arange(start_k, end_k, device=logits.device)

            selected_logits = logits[positions]
            token_ids = torch.tensor(ids, device=logits.device)

            # log_softmax + indexing without full log_softmax
            token_scores = selected_logits.gather(dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)

            log_normalizers = torch.logsumexp(selected_logits.float(), dim=-1)

            target_log_prob = (token_scores.float() - log_normalizers).sum().item()

            target_log_probs.append(target_log_prob)

            del output, logits, selected_logits
            del model_inputs, predicted_ids

        log_probs_list.append(target_log_probs)

    return np.array(log_probs_list)


def create_code_log_probs(
    code_answers: List[str],
    valid_options: List[str],
    high_prob: float = 0.99,
) -> np.ndarray:
    num_rows = len(code_answers)
    num_options = len(valid_options)

    # Calculate probabilities for target vs non-target options
    low_prob = (1.0 - high_prob) / (num_options - 1) if num_options > 1 else 1.0
    
    # Initialize array filled with low probability log-probs
    code_log_probs = np.full((num_rows, num_options), np.log(low_prob))
    
    # Map target option strings to column indices for quick lookup
    option_to_idx = {option: idx for idx, option in enumerate(valid_options)}
    
    # Fill in high log probability for the matching target index
    for i, answer in enumerate(code_answers):
        if answer in option_to_idx:
            code_log_probs[i, option_to_idx[answer]] = np.log(high_prob)
        else:
            # Fallback if answer isn't in valid_options (uniform log prob)
            code_log_probs[i, :] = np.log(1.0 / num_options)
    
    return code_log_probs

def create_alternative_generation_prompts(
    dataset: datasets.arrow_dataset.Dataset,
    task_prompt: str,
    cots: List[str],
) -> List[List[Dict[str, str]]]:
    # Create a new dataset with alternative-generation prompts and corresponding inputs
    gen_dataset = Dataset.from_dict({
        'input': [
            f"Start Of Task Prompt:\n{task_prompt}\nEnd Of Task Prompt\n\n"
            f"Start Of Passage:\n{example['input']}\nEnd Of Passage\n\n"
            f"Start Of Explanation:\n{cot}\nEnd Of Explanation\n\n"
            f"Alter the input passage such that all aspects identified as important in the explanation are preserved, while all other aspects of the input may be changed freely.\n"
            f"First briefly reason about what you can and cannot change about the input passage. Then produce your altered input passage.\n"
            f"The altered input passage should be enclosed exactly between the tags Start Of Passage and End Of Passage."
            for example, cot in zip(dataset, cots)
        ]
    })
    prompts = [
        [
            {'role': 'system', 'content': SCS_PROMPT},
            {'role': 'user', 'content': example['input']},
        ]
        for example in gen_dataset
    ]
    return prompts

def compute_log_sum_exp(
    inputs: List[str],
    log_probs: np.ndarray,
    num_rows: int,
    num_alternatives: int,
) -> np.ndarray:
    log_probs = torch.tensor(log_probs)
    none_mask = torch.tensor([x is None for x in inputs])

    # Reshape inputs and mask to (num_rows, num_alternatives)
    reshaped_log_probs = log_probs.view(num_rows, num_alternatives, -1).clone()
    reshaped_mask = none_mask.view(num_rows, num_alternatives)
    reshaped_log_probs[reshaped_mask] = float('-inf')

    # Count valid alternatives per sample (shape: [num_samples, 1])
    valid_counts = (~reshaped_mask).sum(dim=1, keepdim=True)

    # Compute LogSumExp along the K dimension (dim=1)
    log_sum = torch.logsumexp(reshaped_log_probs, dim=1)

    # Subtract log(valid_counts) for proper probability averaging
    # Use clamp to prevent log(0) for samples where ALL alternatives are None
    combined_log_probs = (log_sum - torch.log(valid_counts.clamp(min=1))).numpy()

    return combined_log_probs

def extract_alt_inputs(
    generations: List[str],
    prefix: str='Start Of Passage',
    suffix: str='End Of Passage', 
) -> List[str]:
    alt_inputs = []
    for gen in generations:
        gen = gen.strip()
        alt_input = None
        input_match = re.search(
            rf'{re.escape(prefix)}[^:\n]*\s*:?\s*(.*?)(?=(?:{re.escape(suffix)})|$)', 
            gen, re.DOTALL
        )
        if not input_match:
            input_match = re.search(
                rf'{re.escape(prefix)}[^:\n]*\s*:\s*(.*?)(?:\n|$)', gen
            )
        if input_match:
            alt_input = input_match.group(1).strip()
        alt_inputs.append(alt_input)
    return alt_inputs
