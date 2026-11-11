from accelerate.utils import set_seed
import argparse
from collections import Counter
from datasets import Dataset
import gc
import json
import numpy as np
import os
import random
import torch

from src.datasets import (
    load_dataset, 
    construct_question_answering_prompts, 
    construct_code_explanation_prompts, 
    construct_refinement_prompts,
)
from src.models import load_llm, generate_outputs
from src.sufficiency import (
    compute_sufficiency, 
    compute_log_probs, 
    create_alternative_generation_prompts,
    compute_log_sum_exp,
    extract_alt_inputs,
    ALTERNATIVE_KWARGS,
)
from src.utils import (
    get_start_time, 
    get_run_time, 
    count_none_frequency, 
    compute_accuracy, 
    extract_chain_of_thought,
    extract_final_answer, 
    match_answer_fuzzy, 
    extract_code_explanation, 
    evaluate_code_explanations,
)
from src.claude_utils import generate_claude_outputs

def parse_args():
    parser = argparse.ArgumentParser(description='LLM Code Explanation')
    parser.add_argument('--dataset-path', type=str, default='lukaemon/bbh')
    parser.add_argument('--dataset-name', type=str, default='web_of_lies')
    parser.add_argument('--model-name', type=str, default='Qwen/Qwen2.5-Coder-32B-Instruct')
    parser.add_argument('--explainer-model-name', type=str, default=None)
    parser.add_argument('--num-examples-per-prompt', type=int, default=20)
    parser.add_argument('--num-explanations-per-dataset', type=int, default=10)
    parser.add_argument('--num-explanations-per-prompt', type=int, default=10)
    parser.add_argument('--return-hidden-states', action='store_true')
    parser.add_argument('--compute-sufficiency', action='store_true')
    parser.add_argument('--code-suff-high-prob', type=float, default=0.99)
    parser.add_argument('--num-text-alternatives', type=int, default=5)
    parser.add_argument('--use-claude', action='store_true')
    parser.add_argument('--agent-max-turns', type=int, default=10)
    parser.add_argument('--agent-max-budget-usd', type=float, default=1.0)
    parser.add_argument('--use-consistency-tool', action='store_true')
    parser.add_argument('--num-refinement-rounds', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--result-file', type=str, default='data/results.json')
    return parser.parse_args()

if __name__ == '__main__':

    ##############################
    ### Load model and dataset ###
    ##############################
    
    # Get start time
    get_start_time()

    # Parse arguments
    args = parse_args()
    if args.use_consistency_tool:
        assert args.use_claude
    if args.explainer_model_name:
        assert not args.compute_sufficiency
    elif args.use_claude or ('claude' in args.model_name):
        assert (
            args.use_claude
            and ('claude' in args.model_name)
            and (not args.explainer_model_name)
            and (not args.return_hidden_states)
            and (not args.compute_sufficiency)
            and (args.num_explanations_per_prompt == 1)
        )
    elif args.num_refinement_rounds > 0:
        assert (
            (not args.explainer_model_name)
            and (not args.return_hidden_states)
            and (not args.compute_sufficiency)
            and (not args.use_claude)
        )

    # Set random seed
    set_seed(args.seed)

    # Load model and tokenizer
    if not args.use_claude:
        print('\nLoading model ...')
        model, tokenizer = load_llm(model_name=args.model_name)

    # Load dataset
    print('\nLoading dataset ...')
    dataset, task_prompt = load_dataset(args.dataset_path, args.dataset_name)
    print('Number of samples:', dataset.num_rows)

    ########################
    ### Get model answer ###
    ########################

    # Construct question-answering prompts
    qa_prompts = construct_question_answering_prompts(dataset, task_prompt)

    # Generate answers
    print('\nGenerating answers ...')
    if args.use_claude:
        generations, qa_costs = generate_claude_outputs(
            prompts=qa_prompts,
            model_name=args.model_name,
        )
    else:
        generation_kwargs = {
            'max_new_tokens': 2000,
            'do_sample': False,
            'temperature': None,
            'top_p': None,
            'top_k': None,
            'repetition_penalty': 1.0,
            'use_cache': True,
        }
        generations = generate_outputs(
            model=model,
            tokenizer=tokenizer,
            prompts=qa_prompts,
            generation_kwargs=generation_kwargs,
            batch_size=4,
        )

        # Extract chain-of-thoughts
        cots = [extract_chain_of_thought(gen) for gen in generations]

    # Extract answers
    extracted_answers = [extract_final_answer(gen) for gen in generations]
    fail_ans_extract = count_none_frequency(extracted_answers)
    print(f'Fail answer extraction: {fail_ans_extract:.4f}')

    # Match answers
    targets = [example['target'] for example in dataset]
    valid_options = np.unique(targets).tolist()
    answers = [match_answer_fuzzy(ans, valid_options) for ans in extracted_answers]
    fail_ans_match = count_none_frequency(answers) - count_none_frequency(extracted_answers)
    print(f'Fail answer matching: {fail_ans_match:.4f}')

    # Compute accuracy
    accuracy = compute_accuracy(answers, targets)
    print(f'Accuracy: {accuracy:.4f}')

    # Add model answer to dataset
    dataset = dataset.map(lambda example, i: {'answer': answers[i]}, with_indices=True)

    valid_indices = [i for i, ans in enumerate(answers) if ans is not None]
    dataset = dataset.select(valid_indices)
    targets = [targets[i] for i in valid_indices]
    answers = [answers[i] for i in valid_indices]
    if not args.use_claude:
        qa_prompts = [qa_prompts[i] for i in valid_indices]
        cots = [cots[i] for i in valid_indices]
    else:
        qa_costs = [qa_costs[i] for i in valid_indices]
    print('Number of samples:', dataset.num_rows)

    # Compute base log probabilities
    if args.compute_sufficiency:
        base_log_probs = compute_log_probs(
            model=model,
            tokenizer=tokenizer,
            prompts=qa_prompts,
            generations=cots,
            valid_options=valid_options,
        )

    #############################
    ### Get model explanation ###
    #############################
    if args.explainer_model_name:
        del model, tokenizer
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        # Load explainer model and tokenizer
        print('\nLoading explainer model ...')
        model, tokenizer = load_llm(model_name=args.explainer_model_name)

    # Split held out train/test subsets
    indices = list(range(dataset.num_rows))
    random.shuffle(indices)
    mid = len(indices) // 2
    train_subset = indices[:mid]
    test_subset = indices[mid:]

    # Compute proportion of majority answer in test subset
    test_answers = [answers[j] for j in test_subset]
    counts = Counter(test_answers)
    _, max_count = counts.most_common(1)[0]
    majority_baseline = max_count / len(test_answers)

    # Get subsets of correct and incorrect answers
    correct_subset = [j for j in test_subset if answers[j] == targets[j]]
    incorrect_subset = [j for j in test_subset if answers[j] != targets[j]]

    # Construct code explanation prompts
    code_explanation_prompts, _ = construct_code_explanation_prompts(
        dataset=dataset, 
        task_prompt=task_prompt, 
        num_examples=args.num_examples_per_prompt, 
        num_explanations=args.num_explanations_per_dataset,
        train_subset=train_subset,
    )

    # Generate code explanations
    print('\nGenerating code explanations ...')
    if args.use_claude:
        code_explanations, code_costs = generate_claude_outputs(
            prompts=code_explanation_prompts,
            model_name=args.model_name,
            code=True,
            max_turns=args.agent_max_turns,
            max_budget_usd=args.agent_max_budget_usd,
            use_consistency_tool=args.use_consistency_tool,
            dataset=dataset,
            train_subset=train_subset,
            answers=answers,
            args=args,
            valid_options=valid_options,
        )
    else:
        CODE_KWARGS = {
            'max_new_tokens': 20000,
            'do_sample': True,
            'temperature': 0.1,
            'top_p': 0.95,
            'repetition_penalty': 1.0,
            'num_return_sequences': args.num_explanations_per_prompt,
            'use_cache': True,
        }
        outputs = generate_outputs(
            model=model,
            tokenizer=tokenizer,
            prompts=code_explanation_prompts,
            generation_kwargs=CODE_KWARGS,
            batch_size=1,
            return_hidden_states=args.return_hidden_states,
        )
        if args.return_hidden_states:
            generations, last_token_hidden_states = outputs
            print('Last token hidden states shape:', last_token_hidden_states.shape)
        else:
            generations = outputs

        # Extract code explanations
        code_explanations = [extract_code_explanation(gen) for gen in generations]

    # Evaluate code explanations
    curr_results = evaluate_code_explanations(
        code_explanations=code_explanations,
        dataset=dataset,
        args=args,
        answers=answers,
        majority_baseline=majority_baseline,
        valid_options=valid_options,
        train_subset=train_subset,
        test_subset=test_subset,
        correct_subset=correct_subset,
        incorrect_subset=incorrect_subset,
        tokenizer=None if args.use_claude else tokenizer,
        base_log_probs=base_log_probs if args.compute_sufficiency else None,
    )

    if args.return_hidden_states:
        curr_results.update({
            'last_token_hidden_states': [lst.tolist() for lst in last_token_hidden_states],
        })
    
    ########################
    ### Save Claude cost ###
    ########################
    if args.use_claude:
        curr_results.update({
            'qa_cost_usd': [float(x) if x is not None else None for x in qa_costs],
            'code_cost_usd': [float(x) if x is not None else None for x in code_costs],
        })
    
    #############################
    ### Multi-turn refinement ###
    #############################
    elif args.num_refinement_rounds > 0:
        refinement_results = [curr_results]
        best_results = curr_results

        for i in range(args.num_refinement_rounds):
            # Construct new code explanation prompts
            refinement_prompts = construct_refinement_prompts(
                code_explanation_prompts=code_explanation_prompts, 
                best_codes=[
                    x['code'] 
                    for x in best_results['code_explanations']
                    if x['cons_train'] == best_results['max_train_cons']
                ],
                max_train_cons=best_results['max_train_cons'],
            )

            # Generate code explanations
            print()
            print('Generating code explanations ...')
            generations = generate_outputs(
                model=model,
                tokenizer=tokenizer,
                prompts=refinement_prompts,
                generation_kwargs=CODE_KWARGS,
                batch_size=1,
                return_hidden_states=False,
            )

            # Extract code explanations
            code_explanations = [extract_code_explanation(gen) for gen in generations]

            # Evaluate new code explanations
            print(f'\n\n===== Iteration {i+1} =====\n\n')
            curr_results = evaluate_code_explanations(
                code_explanations=code_explanations,
                dataset=dataset,
                args=args,
                answers=answers,
                majority_baseline=majority_baseline,
                valid_options=valid_options,
                train_subset=train_subset,
                test_subset=test_subset,
                correct_subset=correct_subset,
                incorrect_subset=incorrect_subset,
                tokenizer=None if args.use_claude else tokenizer,
                base_log_probs=base_log_probs if args.compute_sufficiency else None,
            )
            refinement_results.append(curr_results)

            # Update the best iteration if consistency improved
            if curr_results['max_train_cons'] >= best_results['max_train_cons']:
                best_results = curr_results

        # Print summary of results
        for i, result in enumerate(refinement_results):
            print(
                f'Iter {i}\t|'
                f'Train cons: {result["max_train_cons"]:.4f} |'
                f'Test cons: {result["avg_cons_test"]:.4f} |'
                f'Over majority: {result["cons_over_majority"]:.4f} |'
                f'Chars: {result["avg_num_chars_no_comment"]:.4f}\t|'
                f'CC: {result["avg_cyclomatic_complexity"]:.4f}\n'
            )

    ####################################################
    ### Free-Text Alternative Input Generation & SCS ###
    ####################################################
    elif args.compute_sufficiency:

        print("\nGenerating alternative inputs for free-text explanations...")

        # Setup generating kwargs and SCS prompt template
        alternative_kwargs = {
            **ALTERNATIVE_KWARGS,
            'num_return_sequences': args.num_text_alternatives,
        }
        alt_input_prompts = create_alternative_generation_prompts(
            dataset=dataset, task_prompt=task_prompt, cots=cots
        )

        # Only generate alternatives for the test subset
        alt_input_prompts = [alt_input_prompts[i] for i in test_subset]

        # Generate alternative outputs
        alt_input_generations = generate_outputs(
            model=model,
            tokenizer=tokenizer,
            prompts=alt_input_prompts,
            generation_kwargs=alternative_kwargs,
            batch_size=1,
        )

        # Extract alternative inputs
        alt_inputs = extract_alt_inputs(alt_input_generations)

        # Generate answers using alternative inputs
        alt_qa_prompts = construct_question_answering_prompts(
            dataset=Dataset.from_dict({'input': [str(inp) for inp in alt_inputs]}), 
            task_prompt=task_prompt,
        )

        print("\nGenerating model answers using alternative inputs...")
        alt_generations = generate_outputs(
            model=model,
            tokenizer=tokenizer,
            prompts=alt_qa_prompts,
            generation_kwargs=generation_kwargs,
            batch_size=4,
        )

        # Compute log probabilities for alternative inputs
        alt_cots = [extract_chain_of_thought(gen) for gen in alt_generations]
        alt_log_probs = compute_log_probs(
            model=model,
            tokenizer=tokenizer,
            prompts=alt_qa_prompts,
            generations=alt_cots,
            valid_options=valid_options,
        )

        # Collapse the num_alternatives dimension by computing log-sum-exp across alternatives
        combined_alt_log_probs = compute_log_sum_exp(
            inputs=alt_inputs,
            log_probs=alt_log_probs,
            num_rows=len(test_subset),
            num_alternatives=args.num_text_alternatives,
        )

        # Compute SCSuff
        text_suff_list = compute_sufficiency(base_log_probs[test_subset], combined_alt_log_probs)
        avg_text_suff = np.nanmean(text_suff_list)

        print(f"Average text sufficiency: {avg_text_suff:.4f}")

        text_num_chars_lst = [len(cot) for cot in [cots[i] for i in test_subset]]
        total_text_num_chars = sum(text_num_chars_lst)
        print(f'Number of characters: {total_text_num_chars:.4f}')
        text_num_tokens_lst = [len(tokenizer.encode(cot)) for cot in [cots[i] for i in test_subset]]
        total_text_num_tokens = sum(text_num_tokens_lst)
        print(f'Number of tokens: {total_text_num_tokens:.4f}')

        curr_results.update({
            'avg_text_suff': float(avg_text_suff),
            'total_text_num_chars': int(total_text_num_chars),
            'total_text_num_tokens': int(total_text_num_tokens),
            'text_suff': [float(x) for x in text_suff_list],
            'text_num_chars': [float(x) for x in text_num_chars_lst],
            'text_num_tokens': [float(x) for x in text_num_tokens_lst],
        })

    ##############################
    ### Save code explanations ###
    ##############################
    if os.path.exists(args.result_file):
        with open(args.result_file, 'r') as f:
            data = json.load(f)
    else:
        data = {}
    name = (
        f"dataset_path_{args.dataset_path}-"
        f"dataset_name_{args.dataset_name}-"
        f"model_name_{args.model_name}-"
        f"num_examples_per_prompt_{args.num_examples_per_prompt}-"
        f"num_explanations_per_dataset_{args.num_explanations_per_dataset}-"
        f"num_explanations_per_prompt_{args.num_explanations_per_prompt}-"
        f"num_refinement_rounds_{args.num_refinement_rounds}-"
        f"seed_{args.seed}"
    )
    if args.explainer_model_name:
        name += f"-explainer_model_{args.explainer_model_name}"
    if args.use_consistency_tool:
        name += f'-use_consistency_tool_{args.use_consistency_tool}'

    data[name] = {
        'num_rows': int(dataset.num_rows),
        'answer_acc': float(accuracy),
    }
    if args.num_refinement_rounds > 0:
        data[name]['rounds'] = refinement_results
    else:
        data[name].update(curr_results)

    with open(args.result_file, 'w') as f:
        json.dump(data, f, indent=2)

    # Get runtime
    get_run_time()