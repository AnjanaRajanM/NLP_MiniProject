import boto3

def initialize_bedrock_client():
    """
    Initialize AWS Bedrock client
    """
    return boto3.client(
        service_name="bedrock-runtime",
        region_name="us-east-1"   # CHANGE if needed
    )

import json
from botocore.exceptions import ClientError

def invoke_bedrock_claude(prompt: str, max_tokens: int = 5000, temperature: float = 0.1):
    """
    Invoke Claude model via AWS Bedrock (synchronous)
    Returns parsed JSON or None if 'not a requirement'
    """

    MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    bedrock = initialize_bedrock_client()

    request_payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    try:
        response = bedrock.invoke_model(
            modelId=MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(request_payload).encode("utf-8")
        )

        # Parse
        response_body = json.loads(response["body"].read())
        text = response_body["content"][0]["text"]

        # Clean
        json_str = clean_llm_response(text)
        result = json.loads(json_str)

        # If requirement invalid, skip
        if "not a requirement" in json.dumps(result).lower():
            return None

        return result

    except ClientError as e:
        print(f"AWS Error: Cannot invoke {MODEL_ID}. Reason: {e}")
        return None

    except Exception as e:
        print(f"General Error: {e}")
        return None


import asyncio
import functools
from typing import Any

async def invoke_bedrock_claude_async(prompt: str, max_tokens: int = 5000, temperature: float = 0.1) -> Any:
    """
    Async wrapper around synchronous invoke_bedrock_claude
    """

    loop = asyncio.get_running_loop()

    func = functools.partial(
        invoke_bedrock_claude,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature
    )

    # run sync bedrock invocation in threadpool
    return await loop.run_in_executor(None, func)


import json

def parse_requirement_response(raw_response: str):
    """
    Parse cleaned LLM requirement response into Python object.
    If the requirement is invalid or explicitly 'not a requirement',
    return None so test-case generation can be skipped.
    """

    if raw_response is None:
        return None

    # clean JSON-like string
    json_str = clean_llm_response(raw_response)

    try:
        result = json.loads(json_str)
    except Exception as e:
        print(f"[REQ] JSON parse failed: {e}")
        return None

    # Standard skip logic — REQUIRED for async pipeline:
    text = json.dumps(result).lower()

    if (
        "not a requirement" in text or
        "no valid requirements" in text or
        "cannot classify" in text
    ):
        # return None = skip test-case generation for this chunk
        return None

    return result
