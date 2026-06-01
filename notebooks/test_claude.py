"""
Minimal test that the Anthropic API works and tool-use is functioning.
Set key first:  $env:ANTHROPIC_API_KEY = "sk-ant-..."
"""
import os
from anthropic import Anthropic

client = Anthropic()  # reads ANTHROPIC_API_KEY from env

# 1. basic call
print("=== Test 1: basic call ===")
resp = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=100,
    messages=[{"role": "user", "content": "Say 'API working' and nothing else."}],
)
print(resp.content[0].text)

# 2. tool use - the mechanism the agent relies on
print("\n=== Test 2: tool use ===")
tools = [{
    "name": "get_recommendations",
    "description": "Get book recommendations based on anchor books and filters.",
    "input_schema": {
        "type": "object",
        "properties": {
            "anchor_books": {"type": "array", "items": {"type": "string"},
                             "description": "Titles of books to base recommendations on"},
            "mood": {"type": "string", "description": "Optional mood/tone filter, e.g. 'lighter', 'darker'"},
        },
        "required": ["anchor_books"],
    },
}]

resp = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=300,
    tools=tools,
    messages=[{"role": "user",
               "content": "Recommend me something like The Maltese Falcon and Out, but a bit lighter in tone."}],
)

for block in resp.content:
    if block.type == "text":
        print("TEXT:", block.text)
    elif block.type == "tool_use":
        print("TOOL CALL:", block.name)
        print("INPUT:", block.input)