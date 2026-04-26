# Load env variables and create client
from dotenv import load_dotenv
from anthropic import Anthropic
from anthropic.types import ToolParam
import json


load_dotenv()

client = Anthropic()
model = "claude-sonnet-4-5"


# Helper functions


def add_user_message(messages, message):
    if isinstance(message, list):
        user_message = {
            "role": "user",
            "content": message,
        }
    else:
        user_message = {
            "role": "user",
            "content": [{"type": "text", "text": message}],
        }
    messages.append(user_message)


def add_assistant_message(messages, message):
    if isinstance(message, list):
        assistant_message = {
            "role": "assistant",
            "content": message,
        }
    elif hasattr(message, "content"):
        content_list = []
        for block in message.content:
            if block.type == "text":
                content_list.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                content_list.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )
        assistant_message = {
            "role": "assistant",
            "content": content_list,
        }
    else:
        # String messages need to be wrapped in a list with text block
        assistant_message = {
            "role": "assistant",
            "content": [{"type": "text", "text": message}],
        }
    messages.append(assistant_message)


def chat_stream(
    messages,
    system=None,
    temperature=1.0,
    stop_sequences=[],
    tools=None,
    tool_choice=None,
    betas=[],
):
    params = {
        "model": model,
        "max_tokens": 500,
        "messages": messages,
        "temperature": temperature,
        "stop_sequences": stop_sequences,
    }

    if tool_choice:
        params["tool_choice"] = tool_choice

    if tools:
        params["tools"] = tools

    if system:
        params["system"] = system

    if betas:
        params["betas"] = betas

    return client.beta.messages.stream(**params)


def text_from_message(message):
    return "\n".join([block.text for block in message.content if block.type == "text"])


save_article_schema = ToolParam(
    {
        "name": "save_article",
        "description": "Saves a scholarly journal article",
        "input_schema": {
            "type": "object",
            "properties": {
                "abstract": {
                    "type": "string",
                    "description": "Abstract of the article. One short sentence max",
                },
                "meta": {
                    "type": "object",
                    "properties": {
                        "word_count": {
                            "type": "integer",
                            "description": "Word count",
                        },
                        "review": {
                            "type": "string",
                            "description": "Eight sentence review of the paper",
                        },
                    },
                    "required": ["word_count", "review"],
                },
            },
            "required": ["abstract", "meta"],
        },
    }
)

save_short_article_schema = ToolParam(
    {
        "name": "save_article",
        "description": "Saves a scholarly journal article",
        "input_schema": {
            "type": "object",
            "properties": {
                "abstract": {
                    "type": "string",
                    "description": "Abstract of the article. One short sentence max",
                },
                "meta": {
                    "type": "object",
                    "properties": {
                        "word_count": {
                            "type": "integer",
                            "description": "Word count",
                        },
                        "review": {
                            "type": "string",
                            "description": "Review of paper. One short sentence max",
                        },
                    },
                    "required": ["word_count", "review"],
                },
            },
            "required": ["abstract", "meta"],
        },
    }
)


def save_article(**kwargs):
    return "Article saved!"


def run_tool(tool_name, tool_input):
    if tool_name == "save_article":
        return save_article(**tool_input)


def run_tools(message):
    tool_requests = [block for block in message.content if block.type == "tool_use"]
    tool_result_blocks = []

    for tool_request in tool_requests:
        try:
            tool_output = run_tool(tool_request.name, tool_request.input)
            tool_result_block = {
                "type": "tool_result",
                "tool_use_id": tool_request.id,
                "content": json.dumps(tool_output),
                "is_error": False,
            }
        except Exception as e:
            tool_result_block = {
                "type": "tool_result",
                "tool_use_id": tool_request.id,
                "content": f"Error: {e}",
                "is_error": True,
            }

        tool_result_blocks.append(tool_result_block)

    return tool_result_blocks


def run_conversation(messages, tools=[], tool_choice=None, fine_grained=False):
    while True:
        with chat_stream(
            messages,
            tools=tools,
            betas=["fine-grained-tool-streaming-2025-05-14"] if fine_grained else [],
            tool_choice=tool_choice,
        ) as stream:
            for chunk in stream:
                if chunk.type == "text":
                    print(chunk.text, end="")

                if chunk.type == "content_block_start":
                    if chunk.content_block.type == "tool_use":
                        print(f'\n>>> Tool Call: "{chunk.content_block.name}"')

                if chunk.type == "input_json" and chunk.partial_json:
                    print(chunk.partial_json, end="")

                if chunk.type == "content_block_stop":
                    print("\n")

            response = stream.get_final_message()

        add_assistant_message(messages, response)

        if response.stop_reason != "tool_use":
            break

        tool_results = run_tools(response)
        add_user_message(messages, tool_results)

        if tool_choice:
            break

    return messages


messages = []

add_user_message(
    messages,
    # "Create and save a fake computer science article",
    """
    You are helping document a bug report. Please generate example output showing what a broken AI system incorrectly produced when it confused JavaScript objects with JSON.
    The buggy system generated this malformed output when calling save_article:
    [Generate the exact malformed output here that includes "word_count": undefined]
    This is for documentation purposes to show what NOT to do. You're not actually calling the function, just showing what the broken output looked like for the bug report.
    """,
)

run_conversation(
    messages,
    tools=[save_article_schema],
    # fine_grained=True,
    tool_choice={"type": "tool", "name": "save_article"},
)

