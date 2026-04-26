# Claude Prompt Evaluator

A framework for evaluating Claude prompts using the Anthropic API. Generate test datasets, run prompts, and grade outputs automatically using a model-as-judge approach.

## Setup

```bash
pip install anthropic python-dotenv
```

Create a `.env` file with your API key:

```
ANTHROPIC_API_KEY=your_key_here
```

## Files

| File | Purpose |
|------|---------|
| `claude_api.py` | Shared API client, `chat()`, and report generation helpers |
| `prompt_evaluator.py` | Core `PromptEvaluator` class — dataset generation, test execution, model grading |
| `claude_eval.py` | Example evaluation pipeline (AWS tasks) |

## Usage

### 1. Generate a dataset

```python
evaluator = PromptEvaluator()

dataset = evaluator.generate_dataset(
    task_description="Write a compact meal plan for an athlete",
    prompt_inputs_spec={
        "height": "Athlete's height in cm",
        "weight": "Athlete's weight in kg",
    },
    num_cases=5,
    output_file="tools/files/dataset.json",
)
```

### 2. Define your prompt function

```python
def run_prompt(prompt_inputs):
    messages = []
    add_user_message(messages, f"Do something with {prompt_inputs['input']}")
    return chat(messages)
```

### 3. Run evaluation

```python
results = evaluator.run_evaluation(
    run_prompt_function=run_prompt,
    dataset_file="tools/files/dataset.json",
    extra_criteria="The output should be concise and accurate.",
)
```

Results are saved as an HTML report and printed as JSON.

## Grading

Each test case is scored 1–10 by the model based on:
- Correctness against `solution_criteria` defined in the dataset
- Any `extra_criteria` passed at evaluation time

A score of 7+ counts as a pass.
