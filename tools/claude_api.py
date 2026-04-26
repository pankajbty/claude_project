import os
from statistics import mean

import anthropic
from dotenv import load_dotenv
import json

load_dotenv()
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
model = "claude-sonnet-4-6"


def generate_prompt_evaluation_report(evaluation_results):
    total_tests = len(evaluation_results)
    scores = [result["score"] for result in evaluation_results]
    avg_score = mean(scores) if scores else 0
    max_possible_score = 10
    pass_rate = (
        100 * len([s for s in scores if s >= 7]) / total_tests if total_tests else 0
    )

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Prompt Evaluation Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                color: #333;
            }}
            .header {{
                background-color: #f0f0f0;
                padding: 20px;
                border-radius: 5px;
                margin-bottom: 20px;
            }}
            .summary-stats {{
                display: flex;
                justify-content: space-between;
                flex-wrap: wrap;
                gap: 10px;
            }}
            .stat-box {{
                background-color: #fff;
                border-radius: 5px;
                padding: 15px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                flex-basis: 30%;
                min-width: 200px;
            }}
            .stat-value {{
                font-size: 24px;
                font-weight: bold;
                margin-top: 5px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
            }}
            th {{
                background-color: #4a4a4a;
                color: white;
                text-align: left;
                padding: 12px;
            }}
            td {{
                padding: 10px;
                border-bottom: 1px solid #ddd;
                vertical-align: top;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            .output-cell {{
                white-space: pre-wrap;
            }}
            .score {{
                font-weight: bold;
                padding: 5px 10px;
                border-radius: 3px;
                display: inline-block;
            }}
            .score-high {{
                background-color: #c8e6c9;
                color: #2e7d32;
            }}
            .score-medium {{
                background-color: #fff9c4;
                color: #f57f17;
            }}
            .score-low {{
                background-color: #ffcdd2;
                color: #c62828;
            }}
            .output {{
                overflow: auto;
                white-space: pre-wrap;
            }}

            .output pre {{
                background-color: #f5f5f5;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 10px;
                margin: 0;
                font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                font-size: 14px;
                line-height: 1.4;
                color: #333;
                box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.1);
                overflow-x: auto;
                white-space: pre-wrap; 
                word-wrap: break-word; 
            }}

            td {{
                width: 20%;
            }}
            .score-col {{
                width: 80px;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Prompt Evaluation Report</h1>
            <div class="summary-stats">
                <div class="stat-box">
                    <div>Total Test Cases</div>
                    <div class="stat-value">{total_tests}</div>
                </div>
                <div class="stat-box">
                    <div>Average Score</div>
                    <div class="stat-value">{avg_score:.1f} / {max_possible_score}</div>
                </div>
                <div class="stat-box">
                    <div>Pass Rate (≥7)</div>
                    <div class="stat-value">{pass_rate:.1f}%</div>
                </div>
            </div>
        </div>

        <table>
            <thead>
                <tr>
                    <th>Scenario</th>
                    <th>Prompt Inputs</th>
                    <th>Solution Criteria</th>
                    <th>Output</th>
                    <th>Score</th>
                    <th>Reasoning</th>
                </tr>
            </thead>
            <tbody>
    """

    for result in evaluation_results:
        prompt_inputs_html = "<br>".join(
            [
                f"<strong>{key}:</strong> {value}"
                for key, value in result["test_case"]["prompt_inputs"].items()
            ]
        )

        criteria_string = "<br>• ".join(result["test_case"]["solution_criteria"])

        score = result["score"]
        if score >= 8:
            score_class = "score-high"
        elif score <= 5:
            score_class = "score-low"
        else:
            score_class = "score-medium"

        html += f"""
            <tr>
                <td>{result["test_case"]["scenario"]}</td>
                <td class="prompt-inputs">{prompt_inputs_html}</td>
                <td class="criteria">• {criteria_string}</td>
                <td class="output"><pre>{result["output"]}</pre></td>
                <td class="score-col"><span class="score {score_class}">{score}</span></td>
                <td class="reasoning">{result["reasoning"]}</td>
            </tr>
        """

    html += """
            </tbody>
        </table>
    </body>
    </html>
    """

    return html


def generate_dataset():
    prompt = """
        Generate a evaluation dataset for a prompt evaluation. The dataset will be used to evaluate prompts
        that generate Python, JSON, or Regex specifically for AWS-related tasks. Generate an array of JSON objects,
        each representing task that requires Python, JSON, or a Regex to complete.
        
        Example output:
        ```json
        [
            {
                "task": "Description of task",
                "format": "json" or "python" or "regex"
            },
            ...additional
        ]
        ```
        
        * Focus on tasks that can be solved by writing a single Python function, a single JSON object, or a regular expression.
        * Focus on tasks that do not require writing much code
        
        Please generate 3 objects.
    """

    messages = []
    add_user_message(messages, prompt)
    text = chat(messages)
    # Extract JSON array from the response
    start = text.find("[")
    end = text.rfind("]") + 1
    return json.loads(text[start:end])


def add_user_message(messages, text):
    user_message = {"role": "user", "content": text}
    messages.append(user_message)


def add_assistant_message(messages, text):
    assistant_message = {"role": "assistant", "content": text}
    messages.append(assistant_message)


def chat(messages, system=None, temperature=1.0, stop_sequences=[]):
    params = {
        "model": model,
        "max_tokens": 500,
        "messages": messages,
        "temperature": temperature,
        "stop_sequences": stop_sequences,
    }

    if system:
        params["system"] = system

    message = client.messages.create(**params)
    return message.content[0].text


# def chat(messages, system=None):
#     params = {
#         "model": model,
#         "max_tokens": 1024,
#         "messages": messages
#     }
#
#     if system:
#         params["system"] = system
#
#     response = client.messages.create(**params)
#
#     text = next((b.text for b in response.content if b.type == "text"), "")
#     # print(f"Response: {text}")
#     # print(f"Model: {response.model}")
#     # print(f"Input tokens: {response.usage.input_tokens}")
#     # print(f"Output tokens: {response.usage.output_tokens}")
#     # print(f"Stop reason: {response.stop_reason}")
#     return text


if __name__ == "__main__":
    messages = []

    dataset = generate_dataset()
    with open("files/dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)
    print(dataset)

    # while True:
    #     user_input = input(">>")
    #     print(user_input)
    #     add_user_message(messages, user_input)
    #     answer = chat(messages)
    #     add_assistant_message(messages, answer)
    #     print("--------------------------------")
    #     print(answer)
    #     print("--------------------------------")

    # system = """
    #     You are a patient math tutor.
    #     Do not directly answer a student's questions.
    #     Guide them to a solution step by step.
    # """

    # add_user_message(messages, "How do i solve 5x+3=2 for x?")
    # answer = chat(messages, system)

    # add_assistant_message(messages, answer)
    #
    # add_user_message(messages, "Write another sentence")
    #
    # answer = chat(messages)

    # print(answer)
