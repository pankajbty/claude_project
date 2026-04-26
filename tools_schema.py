import json
import os
import anthropic
from dotenv import load_dotenv
from datetime import datetime, timedelta
from anthropic.types import ToolParam, Message
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

_api_key = os.environ.get("ANTHROPIC_API_KEY")
if not _api_key:
    raise EnvironmentError("ANTHROPIC_API_KEY not found. Check your .env file.")

client = anthropic.Anthropic(api_key=_api_key)
model = "claude-haiku-4-5"


def add_user_message(messages, message):
    user_message = {
        "role": "user",
        "content": message.content if isinstance(message, Message) else message
    }
    messages.append(user_message)


def add_assistant_message(messages, message):
    assistant_message = {
        "role": "assistant",
        "content": message.content if isinstance(message, Message) else message
    }
    messages.append(assistant_message)


def chat(messages, system=None, temperature=1.0, stop_sequences=[], tools=None, tool_choice=None):
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

    message = client.messages.create(**params)
    return message


def text_from_message(message):
    return "\n".join(
        [block.text for block in message.content if block.type == "text"]
    )


# Tools and Schemas
def add_duration_to_datetime(
    datetime_str, duration=0, unit="days", input_format="%Y-%m-%d"
):
    date = datetime.strptime(datetime_str, input_format)

    if unit == "seconds":
        new_date = date + timedelta(seconds=duration)
    elif unit == "minutes":
        new_date = date + timedelta(minutes=duration)
    elif unit == "hours":
        new_date = date + timedelta(hours=duration)
    elif unit == "days":
        new_date = date + timedelta(days=duration)
    elif unit == "weeks":
        new_date = date + timedelta(weeks=duration)
    elif unit == "months":
        month = date.month + duration
        year = date.year + month // 12
        month = month % 12
        if month == 0:
            month = 12
            year -= 1
        day = min(
            date.day,
            [
                31,
                29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28,
                31,
                30,
                31,
                30,
                31,
                31,
                30,
                31,
                30,
                31,
            ][month - 1],
        )
        new_date = date.replace(year=year, month=month, day=day)
    elif unit == "years":
        new_date = date.replace(year=date.year + duration)
    else:
        raise ValueError(f"Unsupported time unit: {unit}")

    return new_date.strftime("%A, %B %d, %Y %I:%M:%S %p")


def set_reminder(content, timestamp):
    print(f"----\nSetting the following reminder for {timestamp}:\n{content}\n----")


def get_current_datetime(date_format="%Y-%m-%d %H:%M:%S"):
    if not date_format:
        raise ValueError("date_format cannot be empty")
    return datetime.now().strftime(date_format)


def run_batch(invocations=[]):
    batch_output = []

    for invocation in invocations:
        name = invocation["name"]
        args = json.loads(invocations["arguments"])

        tool_output = run_tool(name, args)
        batch_output.append({"tool_name": name, "output": tool_output})

    return batch_output


def run_tool(tool_name, tool_input):
    if tool_name == "get_current_datetime":
        return get_current_datetime(**tool_input)
    elif tool_name == "add_duration_to_datetime":
        return add_duration_to_datetime(**tool_input)
    elif tool_name == "set_reminder":
        return set_reminder(**tool_input)
    elif tool_name == "batch_tool":
        return run_batch(**tool_input)


def run_tools(message):
    tool_requests = [
        block for block in message.content if block.type == "tool_use"
    ]
    tool_result_blocks = []

    for tool_request in tool_requests:
        try:
            tool_output = run_tool(tool_request.name, tool_request.input)
            tool_result_block = {
                "type": "tool_result",
                "tool_use_id": tool_request.id,
                "content": json.dumps(tool_output),
                "is_error": False
            }
        except Exception as e:
            tool_result_block = {
                "type": "tool_result",
                "tool_use_id": tool_request.id,
                "content": f"Error: {e}",
                "is_error": True
            }
        tool_result_blocks.append(tool_result_block)

    return tool_result_blocks


def run_conversation(messages):
    while True:
        response = chat(messages, tools=[
            get_current_datetime_schema,
            add_duration_to_datetime_schema,
            set_reminder_schema,
            batch_tool_schema
        ])

        add_assistant_message(messages, response)
        print(text_from_message(response))

        if response.stop_reason != "tool_use":
            break

        tool_results = run_tools(response)
        add_user_message(messages, tool_results)

    return messages


get_current_datetime_schema = ToolParam(
    {
      "name": "get_current_datetime",
      "description": "Returns the current date and time formatted according to the specified format string. Useful when you need to know the current time or date in a specific format.",
      "input_schema": {
        "type": "object",
        "properties": {
          "date_format": {
            "type": "string",
            "description": "A strftime-compatible format string that controls how the datetime is formatted. For example: '%Y-%m-%d' returns '2025-01-30', '%H:%M:%S' returns '14:30:00', '%Y-%m-%d %H:%M:%S' returns '2025-01-30 14:30:00'. Must not be empty.",
            "default": "%Y-%m-%d %H:%M:%S"
          }
        },
        "required": []
      }
    }
)


# messages = []
# add_user_message(messages, "What is the current time in HH:MM format? Also, what is current time in SS format?")
# run_conversation(messages)


# messages = []
# messages.append(
#     {
#         "role": "user",
#         "content": "What is the exact time, formatted as HH:MM:SS?"
#     }
# )
#
# response = client.messages.create(
#     model=model,
#     max_tokens=500,
#     messages=messages,
#     tools=[get_current_datetime_schema],
# )
# print(messages)


# add_user_message(messages, "What is the exact time, formatted as HH:MM:SS?")
# response = client.messages.create(
#     model=model,
#     max_tokens=500,
#     messages=messages,
#     tools=[get_current_datetime_schema],
# )
#
# add_assistant_message(messages, response)
# print(messages)


# messages.append(
#     {
#         "role": "assistant",
#         "content": response.content
#     }
# )
# print(messages)
#
# result = get_current_datetime(**response.content[0].input)
#
# messages.append(
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "tool_result",
#                 "tool_use_id": response.content[0].id,
#                 "content": result,
#                 "is_error": False
#             }
#         ]
#     }
# )
#
# print(messages)
#
# response = client.messages.create(
#     model=model,
#     max_tokens=500,
#     messages=messages,
#     tools=[get_current_datetime_schema],
# )
# print(response)


add_duration_to_datetime_schema = {
    "name": "add_duration_to_datetime",
    "description": "Adds a specified duration to a datetime string and returns the resulting datetime in a detailed format. This tool converts an input datetime string to a Python datetime object, adds the specified duration in the requested unit, and returns a formatted string of the resulting datetime. It handles various time units including seconds, minutes, hours, days, weeks, months, and years, with special handling for month and year calculations to account for varying month lengths and leap years. The output is always returned in a detailed format that includes the day of the week, month name, day, year, and time with AM/PM indicator (e.g., 'Thursday, April 03, 2025 10:30:00 AM').",
    "input_schema": {
        "type": "object",
        "properties": {
            "datetime_str": {
                "type": "string",
                "description": "The input datetime string to which the duration will be added. This should be formatted according to the input_format parameter.",
            },
            "duration": {
                "type": "number",
                "description": "The amount of time to add to the datetime. Can be positive (for future dates) or negative (for past dates). Defaults to 0.",
            },
            "unit": {
                "type": "string",
                "description": "The unit of time for the duration. Must be one of: 'seconds', 'minutes', 'hours', 'days', 'weeks', 'months', or 'years'. Defaults to 'days'.",
            },
            "input_format": {
                "type": "string",
                "description": "The format string for parsing the input datetime_str, using Python's strptime format codes. For example, '%Y-%m-%d' for ISO format dates like '2025-04-03'. Defaults to '%Y-%m-%d'.",
            },
        },
        "required": ["datetime_str"],
    },
}

set_reminder_schema = {
    "name": "set_reminder",
    "description": "Creates a timed reminder that will notify the user at the specified time with the provided content. This tool schedules a notification to be delivered to the user at the exact timestamp provided. It should be used when a user wants to be reminded about something specific at a future point in time. The reminder system will store the content and timestamp, then trigger a notification through the user's preferred notification channels (mobile alerts, email, etc.) when the specified time arrives. Reminders are persisted even if the application is closed or the device is restarted. Users can rely on this function for important time-sensitive notifications such as meetings, tasks, medication schedules, or any other time-bound activities.",
    "input_schema": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The message text that will be displayed in the reminder notification. This should contain the specific information the user wants to be reminded about, such as 'Take medication', 'Join video call with team', or 'Pay utility bills'.",
            },
            "timestamp": {
                "type": "string",
                "description": "The exact date and time when the reminder should be triggered, formatted as an ISO 8601 timestamp (YYYY-MM-DDTHH:MM:SS) or a Unix timestamp. The system handles all timezone processing internally, ensuring reminders are triggered at the correct time regardless of where the user is located. Users can simply specify the desired time without worrying about timezone configurations.",
            },
        },
        "required": ["content", "timestamp"],
    },
}

batch_tool_schema = {
    "name": "batch_tool",
    "description": "Invoke multiple other tool calls simultaneously",
    "input_schema": {
        "type": "object",
        "properties": {
            "invocations": {
                "type": "array",
                "description": "The tool calls to invoke",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "The name of the tool to invoke",
                        },
                        "arguments": {
                            "type": "string",
                            "description": "The arguments to the tool, encoded as a JSON string",
                        },
                    },
                    "required": ["name", "arguments"],
                },
            }
        },
        "required": ["invocations"],
    },
}

article_summary_schema = {
    "name": "article_summary",
    "description": "Creates a summary of an article with its key insights. Use this tool when you need to generate a structured summary of an article, research paper, or any textual content. The tool requires the article's title, author name, and a list of the most important insights or takeaways from the content. Each insight should be a concise statement capturing a significant point from the article.",
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "The title of the article being summarized.",
            },
            "author": {
                "type": "string",
                "description": "The name of the author who wrote the article.",
            },
            "key_insights": {
                "type": "array",
                "items": {"type": "string"},
                "description": "A list of the most important takeaways or insights from the article. Each insight should be a complete, concise statement.",
            },
        },
        "required": ["title", "author", "key_insights"],
    },
}


# messages = []
# add_user_message(messages, """
#                  Set two reminder for Jan 1st 2025 at 8AM.
#                     * I have a doctor's appointment
#                     * taxes are due
#                  """
#                  )
# run_conversation(messages)
# print(messages)


messages = []
add_user_message(messages, """
                 Write a one-paragraph scholarly article about computer science.
                 Include a title and author name.
                 """
                 )
response = chat(messages)
result = text_from_message(response)

add_user_message(messages, text_from_message(response))
response = chat(messages, tools=[article_summary_schema], tool_choice={"type": "tool", "name": "article_summary"})
print(response.content[0].input)
