def clean_llm_response(response):
    """
    Normalize Bedrock Claude response into a clean JSON string.
    Handles list, dict, and raw string formats.
    """

    # If response is a list of content blocks → join them
    if isinstance(response, list):
        response = "".join(
            block.get("text", "")
            if isinstance(block, dict)
            else str(block)
            for block in response
        )

    # If response is a Bedrock dictionary
    if isinstance(response, dict):
        # Claude's response is usually inside response["content"][0]["text"]
        if "content" in response and isinstance(response["content"], list):
            response = "".join(
                item.get("text", "")
                for item in response["content"]
                if isinstance(item, dict)
            )
        else:
            # fallback
            response = str(response)

    # Now ensure it is a string
    if not isinstance(response, str):
        response = str(response)

    response = response.strip()

    # Strip unwanted markdown or text around JSON
    # Try to extract first valid JSON object or array
    start = response.find("{")
    end = response.rfind("}") + 1

    if start != -1 and end != -1 and end > start:
        return response[start:end]

    start = response.find("[")
    end = response.rfind("]") + 1

    if start != -1 and end != -1 and end > start:
        return response[start:end]

    # If nothing valid found → return cleaned plain text
    return response
