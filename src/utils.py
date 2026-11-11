import ast
from datasets.arrow_dataset import Dataset
import numpy as np
import os
from radon.complexity import cc_visit
import re
import subprocess
import tempfile
import time
from tqdm import trange
from transformers import AutoTokenizer

from .datasets import process_input
from .sufficiency import create_code_log_probs, compute_sufficiency

TIME = None

def get_start_time():
    global TIME
    TIME = time.time()

def get_run_time():
    global TIME
    if TIME is None:
        raise ValueError('Please call get_start_time() before calling get_run_time().')
    print(f'\nRuntime: {(time.time() - TIME)/60:.2f} minutes')

def count_none_frequency(lst: list[str]) -> float:
    return np.array([x is None for x in lst]).mean()

def compute_accuracy(preds: list[str], targets: list[str]) -> float:
    assert len(preds) == len(targets)
    return np.array([str(y_pred) == str(y_true) for y_pred, y_true in zip(preds, targets)]).mean()

def extract_final_answer(text: str, default_full_text: bool = False) -> str:
    pattern = r'FINAL_ANSWER>?:?\s*(.*?)\s*<?/?FINAL_ANSWER'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    pattern = r'([^\n<>]+?)\s*<?/?FINAL_ANSWER'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    pattern = r'FINAL_ANSWER>?:?\s*(.+)'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    if default_full_text:
        return text
    return None

def extract_chain_of_thought(text: str) -> str:
    pattern = r'<?/?FINAL_ANSWER>?:?'
    match = re.search(pattern, text)
    if match:
        return text[:match.start()].strip()
    return text.strip()

def normalize(text: str) -> str:
    return re.sub(r'[^\w\s]', '', text).strip().lower()

def match_answer_fuzzy(text: str, valid_options: list[str]) -> str:
    if text is None:
        return None
    text_norm = normalize(text)
    for option in valid_options:
        if normalize(option) == text_norm:
            return option
    return None

def extract_code_explanation(text: str) -> str:
    pattern = r'FINAL_ANSWER>?\s*```python\s*(.*?)\s*```\s*<?/?FINAL_ANSWER'
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    pattern = r'```python\s*(.*?)\s*```'
    match = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match[-1].strip()
    
    pattern = r'FINAL_ANSWER>?\s*```python\s*(.*?)\s*\s*<?/?FINAL_ANSWER'
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    pattern = r'```python\s*([\s\S]*)$'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    return  None

def is_valid_python(code: str) -> bool:
    if not isinstance(code, str):
        return False
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False

def run_code_safe(code: str, x) -> str:
    if isinstance(x, tuple) or isinstance(x, list):
        x = tuple(x)
        script = f'''
import sys
{code}
print(solve{repr(x)}, end='')
'''
    else:
        script = f'''
import sys
{code}
print(solve({repr(x)}), end='')
'''
    try:
        out = subprocess.run(
            ['python3', '-c', script],
            capture_output=True,
            timeout=1
        )
        if out.returncode != 0:
            return None
        return out.stdout.decode()
    except Exception:
        return None

def get_pylint_score(code: str) -> float:
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(code)
        temp_path = f.name

    try:
        result = subprocess.run(
            ["pylint", temp_path, "--score=y"],
            capture_output=True,
            text=True
        )
        # Extract score from output
        for line in result.stdout.splitlines():
            if "rated at" in line:
                score = float(line.split("rated at")[1].split("/")[0])
                return score
        return None
    finally:
        os.remove(temp_path)

def get_cyclomatic_complexity(code: str) -> float:
    blocks = cc_visit(code)
    return sum(block.complexity for block in blocks)

def evaluate_code_explanations(
    code_explanations: list[str],
    dataset: Dataset,
    args,
    answers: list[str],
    majority_baseline: float,
    valid_options: list[str],
    train_subset: list[int],
    test_subset: list[int],
    correct_subset: list[int],
    incorrect_subset: list[int],
    tokenizer: AutoTokenizer = None,
    base_log_probs: np.ndarray = None,
):
    """Evaluate each iteration of code explanations"""
    # Check for invalid python code
    fail_code_extract = count_none_frequency(code_explanations)
    print(f'Fail code extraction: {fail_code_extract:.4f}')
    invalid_code = np.mean([
        code is not None and not is_valid_python(code)
        for code in code_explanations
    ])
    print(f'Invalid Python code: {invalid_code:.4f}')

    # Evaluate code explanations one-by-one
    print('\nEvaluating code explanations ...')

    # Initialize lists to store code consistencies
    code_consistencies_train = []
    code_consistencies_test = []
    code_consistencies_correct = []
    code_consistencies_incorrect = []
    if args.compute_sufficiency:
        code_sufficiency = []
        code_suff_list = []

    for i in trange(len(code_explanations)):
        code = code_explanations[i]

        # Generate code answers
        try:
            if not code or not is_valid_python(code):
                raise ValueError('Missing or invalid code')
            multiple_inputs = ',' in code.split('def solve(')[1].split('):')[0]
        except (AttributeError, IndexError, ValueError):
            code_answers = [None] * dataset.num_rows
        else:
            code_answers = [
                run_code_safe(code, process_input(
                    example['input'], 
                    dataset_path=args.dataset_path, 
                    dataset_name=args.dataset_name,
                    multiple_inputs=multiple_inputs,
                )) 
                for example in dataset
            ]

        # Match answers with valid options
        code_answers = [
            match_answer_fuzzy(
                extract_final_answer(str(ans), default_full_text=True), 
                valid_options,
            ) 
            for ans in code_answers
        ]

        # Compute code consistency
        cons_train = compute_accuracy(
            [code_answers[j] for j in train_subset], [answers[j] for j in train_subset]
        )
        code_consistencies_train.append(cons_train)

        cons_test = compute_accuracy(
            [code_answers[j] for j in test_subset], [answers[j] for j in test_subset]
        )
        code_consistencies_test.append(cons_test)

        cons_correct = compute_accuracy(
            [code_answers[j] for j in correct_subset], [answers[j] for j in correct_subset]
        )
        code_consistencies_correct.append(cons_correct)

        cons_incorrect = compute_accuracy(
            [code_answers[j] for j in incorrect_subset], [answers[j] for j in incorrect_subset]
        )
        code_consistencies_incorrect.append(cons_incorrect)

        # Compute code sufficiency
        if args.compute_sufficiency:
            code_log_probs = create_code_log_probs(
                code_answers, valid_options, high_prob=args.code_suff_high_prob
            )
            code_suff_lst = compute_sufficiency(
                base_log_probs[test_subset], code_log_probs[test_subset]
            )
            avg_code_suff = np.nanmean(code_suff_lst)
            code_sufficiency.append(avg_code_suff)
            code_suff_list.append(code_suff_lst)

    code_consistencies_train = np.array(code_consistencies_train)
    code_consistencies_test = np.array(code_consistencies_test)
    code_consistencies_correct = np.array(code_consistencies_correct)
    code_consistencies_incorrect = np.array(code_consistencies_incorrect)
    if args.compute_sufficiency:
        code_sufficiency = np.array(code_sufficiency)

    # Get all best code
    max_train_cons = code_consistencies_train.max()
    best_codes = []
    best_cons_test = []
    best_cons_correct = []
    best_cons_incorrect = []
    if args.compute_sufficiency:
        best_avg_code_suff = []
        best_code_suff_list = []

    for i in range(len(code_explanations)):
        if code_consistencies_train[i] == max_train_cons:
            best_codes.append(code_explanations[i])
            best_cons_test.append(code_consistencies_test[i])
            best_cons_correct.append(code_consistencies_correct[i])
            best_cons_incorrect.append(code_consistencies_incorrect[i])
            if args.compute_sufficiency:
                best_avg_code_suff.append(code_sufficiency[i])
                best_code_suff_list.append(code_suff_list[i])

    # Compute relevant metrics
    print(f'\nMax code train consistency: {max_train_cons:.4f}')
    
    num_code_selected = len(best_codes)
    print(f'Number of code explanations selected: {num_code_selected}')
    
    avg_cons_test = sum(best_cons_test) / num_code_selected
    print(f'Average code test consistency: {avg_cons_test:.4f}')
    
    avg_cons_correct = sum(best_cons_correct) / num_code_selected
    print(f'Average code consistency for correct answers: {avg_cons_correct:.4f}')
    
    avg_cons_incorrect = sum(best_cons_incorrect) / num_code_selected
    print(f'Average code consistency for incorrect answers: {avg_cons_incorrect:.4f}')

    print(f'Majority baseline for test subset: {majority_baseline:.4f}')

    cons_over_majority = avg_cons_test - majority_baseline
    print(f'Code consistency over majority baseline: {cons_over_majority:.4f}')

    k = args.num_explanations_per_prompt
    best_indices = code_consistencies_train.reshape(-1, k).argmax(axis=1)
    test_cons = code_consistencies_test.reshape(-1, k)
    best_of_k_cons = test_cons[np.arange(len(best_indices)), best_indices]
    print(f'Best-of-{k} code consistency list: {best_of_k_cons}')

    if args.compute_sufficiency:
        avg_code_suff = sum(best_avg_code_suff) / num_code_selected
        print(f'Average code sufficiency: {avg_code_suff:.4f}')

    # Compute length metrics
    num_lines_no_comment_no_empty_lst = [
        sum(1 for line in code.splitlines() if line.split('#', 1)[0].strip() != "") 
        for code in best_codes
    ]
    num_lines_no_comment_no_empty = sum(num_lines_no_comment_no_empty_lst) / num_code_selected
   
    num_chars_no_comment_no_empty_lst = [
        sum(
            sum(not c.isspace() for c in line.split('#', 1)[0]) 
            for line in code.splitlines()
        )
        for code in best_codes 
    ]
    num_chars_no_comment_no_empty = sum(num_chars_no_comment_no_empty_lst) / num_code_selected
    print(f'Average number of characters: {num_chars_no_comment_no_empty:.4f}')
    
    if tokenizer:
        num_tokens_lst = [len(tokenizer.encode(code)) for code in best_codes]
        num_tokens = sum(num_tokens_lst) / num_code_selected
        print(f'Average number of tokens: {num_tokens:.4f}')
    
    pylint_score_lst = [get_pylint_score(code) for code in best_codes]
    pylint_score = sum(pylint_score_lst) / num_code_selected
    cyclomatic_complexity_lst = [get_cyclomatic_complexity(code) for code in best_codes]
    cyclomatic_complexity = sum(cyclomatic_complexity_lst) / num_code_selected

    results = {
        'max_train_cons': float(max_train_cons),
        'num_code_selected': int(num_code_selected),
        'avg_cons_test': float(avg_cons_test),
        'avg_cons_correct': float(avg_cons_correct),
        'avg_cons_incorrect': float(avg_cons_incorrect),
        'majority_baseline': float(majority_baseline),
        'cons_over_majority': float(cons_over_majority),
        'best_of_k_cons': [float(x) for x in best_of_k_cons],
        'avg_num_lines_no_comment': float(num_lines_no_comment_no_empty),
        'avg_num_chars_no_comment': float(num_chars_no_comment_no_empty),
        'avg_pylint_score': float(pylint_score),
        'avg_cyclomatic_complexity': float(cyclomatic_complexity),
        'num_lines_no_comment': [float(x) for x in num_lines_no_comment_no_empty_lst],
        'num_chars_no_comment': [float(x) for x in num_chars_no_comment_no_empty_lst],
        'pylint_score': [float(x) for x in pylint_score_lst],
        'cyclomatic_complexity': [float(x) for x in cyclomatic_complexity_lst],
        'code_explanations': [
            {
                'cons_train': float(cons_train),
                'cons_test': float(cons_test),
                'cons_correct': float(cons_correct),
                'cons_incorrect': float(cons_incorrect),
                'code': code,
            }
            for cons_train, cons_test, cons_correct, cons_incorrect, code in zip(
                code_consistencies_train, code_consistencies_test, 
                code_consistencies_correct, code_consistencies_incorrect,
                code_explanations
            )
        ],
    }
    if tokenizer:
        results.update({
            'avg_num_tokens': float(num_tokens),
            'num_tokens': [float(x) for x in num_tokens_lst],
        })
    if args.compute_sufficiency:
        results.update({
            'avg_code_suff': float(avg_code_suff),
            'code_suff': [[float(x) for x in code_row] for code_row in best_code_suff_list],
        })

    return results