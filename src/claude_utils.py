import asyncio
import json
from pathlib import Path
import tempfile
from tqdm import trange

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ResultMessage,
    ResultError,
    query,
    tool,
    create_sdk_mcp_server,
)

from .datasets import process_input
from .utils import (
    compute_accuracy,
    extract_final_answer,
    is_valid_python,
    match_answer_fuzzy,
    run_code_safe,
)

# Use Claude for ordinary answers or code explanations.
def generate_claude_outputs(
    prompts, model_name, code=False, max_turns=10, max_budget_usd=1.0,
    use_consistency_tool=False, dataset=None, train_subset=None,
    answers=None, args=None, valid_options=None,
):
    async def run():
        generations, costs = [], []

        for i in trange(len(prompts)):
            # Preserve the original system/user roles
            messages = prompts[i]
            system_prompt = messages[0]['content']
            user_prompt = messages[1]['content']

            if not code:
                options = ClaudeAgentOptions(
                    model=model_name,
                    system_prompt=system_prompt,
                    tools=[],
                    max_turns=1,
                    setting_sources=[],
                )

                result = None
                async for message in query(
                    prompt=user_prompt,
                    options=options,
                ):
                    if isinstance(message, ResultMessage):
                        result = message

                if result is None or result.is_error:
                    raise RuntimeError(f'QA failed on prompt {i}: {result}')

                generations.append(result.result or '')
                costs.append(result.total_cost_usd)

            else:
                with tempfile.TemporaryDirectory() as work_dir:
                    allowed_tools = ['Bash', 'Read', 'Write', 'Edit']
                    mcp_servers = {}

                    if use_consistency_tool:
                        evaluation_server = create_consistency_tool(
                            work_dir=work_dir,
                            dataset=dataset,
                            train_subset=train_subset,
                            answers=answers,
                            args=args,
                            valid_options=valid_options,
                        )

                        mcp_servers['evaluation'] = evaluation_server
                        allowed_tools.append('mcp__evaluation__compute_consistency')

                    options = ClaudeAgentOptions(
                        model=model_name,
                        system_prompt=system_prompt,
                        cwd=work_dir,
                        tools=['Bash', 'Read', 'Write', 'Edit'],
                        allowed_tools=allowed_tools,
                        mcp_servers=mcp_servers,
                        permission_mode='dontAsk',
                        setting_sources=[],
                        max_turns=max_turns,
                        max_budget_usd=max_budget_usd,
                    )

                    instruction = (
                        "\n\nYou may write, run, test, and revise your Python code using the available tools. "
                        "Save the final code in solve.py in the current working directory. "
                        "The file must define a solve(...) function. "
                        "Test your code before finishing. "
                        "Do not access external datasets."
                    )

                    if use_consistency_tool:
                        instruction += (
                            "\n\nYou also have a compute_consistency tool. "
                            "After writing solve.py, call this tool to measure how consistently your program reproduces your original answers on the training set. "
                            "You may revise solve.py and call the tool again. "
                            "Aim to maximize consistency. "
                            "Before finishing, ensure solve.py contains the best program you found according to training consistency. "
                        )

                    result = None

                    try:
                        async for message in query(
                            prompt=user_prompt + instruction,
                            options=options,
                        ):
                            if isinstance(message, ResultMessage):
                                result = message

                    except ResultError as e:
                        if e.subtype == 'error_max_turns' or e.terminal_reason == 'max_turns':
                            print(f'Warning: Code run {i} reached max turns ({max_turns})')
                        else:
                            raise

                    path = Path(work_dir) / 'solve.py'
                    generations.append(
                        path.read_text(encoding='utf-8')
                        if path.exists() else None
                    )
                    costs.append(
                        result.total_cost_usd if result else None
                    )

                    if result is None or result.is_error:
                        print(f'Warning: code run {i} failed: {result}')

        return generations, costs

    return asyncio.run(run())


# Code consistency tool
def create_consistency_tool(work_dir, dataset, train_subset, answers, args, valid_options):
    @tool(
        "compute_consistency",
        "Execute solve.py on the training set and return the fraction of outputs matching your original answers. ",
        {},
    )
    async def compute_consistency_tool(params):
        path = Path(work_dir) / 'solve.py'

        if not path.is_file():
            result = {
                'consistency': 0.0,
                'error': 'solve.py does not exist',
            }
        else:
            code = path.read_text(encoding='utf-8')

            if not is_valid_python(code):
                result = {
                    'consistency': 0.0,
                    'error': 'Invalid Python code',
                }
            else:
                try:
                    multiple_inputs = (
                        ',' in code.split('def solve(')[1].split('):')[0]
                    )

                    code_answers = [
                        run_code_safe(
                            code,
                            process_input(
                                dataset[j]['input'],
                                dataset_path=args.dataset_path,
                                dataset_name=args.dataset_name,
                                multiple_inputs=multiple_inputs,
                            ),
                        )
                        for j in train_subset
                    ]

                    code_answers = [
                        match_answer_fuzzy(
                            extract_final_answer(
                                str(ans), default_full_text=True
                            ),
                            valid_options,
                        )
                        for ans in code_answers
                    ]

                    expected = [answers[j] for j in train_subset]
                    consistency = compute_accuracy(code_answers, expected)

                    result = {
                        'consistency': float(consistency),
                    }

                except Exception as exc:
                    result = {
                        'consistency': 0.0,
                        'error': str(exc),
                    }

        return {
            'content': [
                {
                    'type': 'text',
                    'text': json.dumps(result),
                }
            ]
        }

    return create_sdk_mcp_server(
        name='evaluation',
        version='1.0.0',
        tools=[compute_consistency_tool],
    )