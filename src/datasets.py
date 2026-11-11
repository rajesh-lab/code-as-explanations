from copy import deepcopy
import datasets
import numpy as np
import os
import random
from typing import Tuple, Dict, List

# Load dataset
def load_dataset(
    dataset_path: str, dataset_name: str, split: str = 'test'
) -> Tuple[datasets.arrow_dataset.Dataset, str]:
    dataset = datasets.load_dataset(
        path=dataset_path,
        name=dataset_name,
        split=split,
        cache_dir=os.environ['HF_HOME'],
    )
    if dataset_path == 'lukaemon/bbh':
        task_prompt = 'You are a helpful assistant. Answer the following question. Solve the problem step by step. Then give the final answer in the format <FINAL_ANSWER>\nanswer\n</FINAL_ANSWER>. If answer choices are labeled with letters, output only the single letter corresponding to the correct choice in the format <FINAL_ANSWER>\nletter\n</FINAL_ANSWER>. If the question is a Yes or No question, output only "Yes" or "No" in the format <FINAL_ANSWER>\nYes/No\n</FINAL_ANSWER>. It\'s very important that you stick to this format.'
    else:
        raise ValueError('Missing task prompt')
    return dataset, task_prompt

# Construct question-answering prompts
def construct_question_answering_prompts(
    dataset: datasets.arrow_dataset.Dataset, task_prompt: str,
) -> List[List[Dict[str, str]]]:
    prompts = [
        [
            {'role': 'system', 'content': task_prompt},
            {'role': 'user', 'content': 'Q: ' + example['input']},
        ]
        for example in dataset
    ]
    return prompts


# Construct code explanation prompts
def construct_code_explanation_prompts(
    dataset: datasets.arrow_dataset.Dataset, task_prompt: str, num_examples: int = 20, 
    num_explanations: int = 10, train_subset: List = None,
) -> Tuple[List[List[Dict[str, str]]], List[int]]:
    system_prompt = f'You are given a task along with {num_examples} example input-output pairs that you previously answered. Your task is to explain your underlying algorithm. The function must correctly handle ALL tokens that appear in the inputs. Do not ignore any part of the input string.'
    if train_subset is None:
        train_subset = dataset.num_rows
    input_subsets = [
        np.random.choice(train_subset, size=num_examples, replace=False).tolist()
        for _ in range(num_explanations)
    ]
    prompts = [
        [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': (
                f'Task description:\n{task_prompt}\n\nExamples:\n'
                + '\n'.join([
                    f'Q: {example["input"]}\nA: {example["answer"]}\n'
                    for example in dataset.select(subset)
                ])
                + 'Write a Python function that reproduces the outputs from the inputs. Think about how to write an algorithm step by step. Make sure that your code does not raise any error when running with the input strings. Do not show example usage. Output the final Python code in the following format:\n<FINAL_ANSWER>\n```python\ndef solve(x):\t...\n```\n</FINAL_ANSWER>'
            )},
        ]
        for subset in input_subsets
    ]
    return prompts, input_subsets


# Construct iterative refinement prompt
def construct_refinement_prompts(
    code_explanation_prompts: List[List[Dict[str, str]]], best_codes: List[str], max_train_cons: float
) -> List[List[Dict[str, str]]]:
    code_examples = random.choices(best_codes, k=len(code_explanation_prompts))
    new_prompts = deepcopy(code_explanation_prompts)
    for prompt, code in zip(new_prompts, code_examples):
        refinement_instruction = (
            f'\n\nYou previously produced this Python code explanation:\n```python\n{code}\n```\n'
            + f'Your previous code explanation achieved a consistency score of {max_train_cons:.4f}, '
            + 'measured as the fraction of inputs on which the code explanation\'s outputs match your original answers. The goal is 1.0 (perfect agreement).\n'
            + 'Identify potential errors and revise your code to improve consistency. Follow the original instructions and return the complete, executable Python code.'
        )
        assert prompt[-1]['role'] == 'user'
        prompt[-1]['content'] += refinement_instruction
    return new_prompts


# Process input
def process_input(
    input_string: str, dataset_path: str, dataset_name: str, multiple_inputs: bool = False
):
    if dataset_path == 'lukaemon/bbh':
        if dataset_name == 'boolean_expressions':
            return input_string.replace(' is', '')
        if dataset_name == 'multistep_arithmetic_two':
            return input_string.replace(' =', '')
        if dataset_name == 'web_of_lies':
            return input_string.replace('Question: ', '')
        if dataset_name == 'formal_fallacies':
            return input_string.replace('\nIs the argument, given the explicitly stated premises, deductively valid or invalid?\nOptions:\n- valid\n- invalid', '')
        if dataset_name == 'salient_translation_error_detection':
            if multiple_inputs:
                source = input_string.split('Source: ')[1].split('\n')[0]
                translation = input_string.split('Translation: ')[1].split('\n')[0]
                return source, translation
            return input_string.replace('The following translations from German to English contain a particular error. That error will be one of the following types: Named Entities: An entity (names, places, locations, etc.) is changed to a different entity. Numerical Values: Numerical values (ordinals or cardinals), dates, and/or units are changed. Modifiers or Adjectives: The modifiers and adjectives pertaining to a noun are changed. Negation or Antonyms: Introduce or remove a negation or change comparatives to their antonyms. Facts: Trivial factual errors not pertaining to the above classes are introduced in the translations. Dropped Content: A significant clause in the translation is removed. Please identify that error. ', '')
        if dataset_name == 'disambiguation_qa':
            if multiple_inputs:
                sentence = input_string.split('Sentence: ')[1].split('\n')[0]
                options = [
                    x.replace('(', '') 
                    for x in input_string.split('Options:\n')[1].split('\n')
                ]
                return sentence, options
            return input_string.replace('In the following sentences, explain the antecedent of the pronoun (which thing the pronoun refers to), or state that it is ambiguous.\n', '')
    return input_string