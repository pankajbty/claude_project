from statistics import mean
from dotenv import load_dotenv
import json
import re
import ast

from claude_api import add_user_message, chat

load_dotenv()


def validate_json(text):
    try:
        json.loads(text.strip())
        return 10
    except json.JSONDecodeError:
        return 0


def validate_python(text):
    try:
        ast.parse(text.strip())
        return 10
    except SyntaxError:
        return 0


def validate_regex(text):
    try:
        re.compile(text.strip())
        return 10
    except re.error:
        return 0


def grade_syntax(response, test_case):
    format = test_case["format"]
    if format == "json":
        return validate_json(response)
    elif format == "python":
        return validate_python(response)
    else:
        return validate_regex(response)


def run_prompt(test_case):
    prompt = f"""Please solve the following task: {test_case['task']}
     * Respond only with the raw Python, JSON, or plain Regex — no markdown, no code blocks, no explanation
    """
    messages = []
    add_user_message(messages, prompt)
    output = chat(messages)
    return output


def grade_by_model(test_case, output):
    eval_prompt = f"""
        You are an expert AWS code reviewer. Your task is to evaluate the following AI-generated solution.
        
        Original Task:
        <task>
        {test_case["task"]}
        </task>
        
        Solution to Evaluate:
        <solution>
        {output}
        </solution>
        
        Output Format
        Provide your evaluation as a structured JSON object with the following fields, in this specific order:
        - "strengths": An array of 1-3 key strengths
        - "weaknesses": An array of 1-3 key areas for improvement
        - "reasoning": A concise explanation of your overall assessment
        - "score": A number between 1-10
        
        Respond with raw JSON only — no markdown, no code blocks, no explanation. Keep your response concise and direct.
        Example response shape:
        {{
            "strengths": string[],
            "weaknesses": string[],
            "reasoning": string,
            "score": number
        }}
    """

    messages = []
    add_user_message(messages, eval_prompt)
    eval_text = chat(messages)
    # Extract JSON object from the response
    start = eval_text.find("{")
    end = eval_text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in model response:\n{eval_text}")
    return json.loads(eval_text[start:end])


def run_test_case(test_case):
    output = run_prompt(test_case)

    # TODO- Grading
    model_grade = grade_by_model(test_case, output)
    model_score = model_grade["score"]
    reasoning = model_grade["reasoning"]

    syntax_score = grade_syntax(output, test_case)
    score = (model_score + syntax_score) /2

    return {
        "output": output,
        "test_case": test_case,
        "score": score,
        "reasoning": reasoning
    }


def run_eval(dataset):
    results = []
    for test_case in dataset:
        result = run_test_case(test_case)
        results.append(result)

    average_score = mean([result["score"] for result in results])
    print(f"Average score: {average_score}")

    return results


with open("dataset.json", "r") as f:
    dataset = json.load(f)

results = run_eval(dataset)

print(json.dumps(results, indent=2))
