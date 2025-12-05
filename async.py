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

import json

def parse_testcase_response(raw_response: str):
    """
    Parse cleaned LLM test-case response into Python object.
    If the test-case JSON is invalid, return None to trigger retry logic.
    """

    if raw_response is None:
        return None

    json_str = clean_llm_response(raw_response)

    try:
        result = json.loads(json_str)
    except Exception as e:
        print(f"[TC] JSON parse failed: {e}")
        return None

    # Validate using your existing validation logic
    try:
        validate_test_cases(result)   # this MUST exist already
    except Exception as e:
        print(f"[TC] Validation failed: {e}")
        return None

    return result



import json

async def generate_requirement_for_chunk(chunk, chunk_num, total_chunks):
    """
    Generate requirement(s) for a single chunk of document text.
    Returns:
        requirement_obj (Python dict/list)
        or None → skip test-case generation
    """

    try:
        # Build requirement prompt (uses your existing global template)
        prompt = prompt_requirements_extraction.format(
            requirements_structure=requirements_structure,
            chunk=chunk,
            chunk_num=chunk_num,
            total_chunks=total_chunks
        )

        # Async Claude call
        raw_response = await invoke_bedrock_claude_async(
            prompt=prompt,
            max_tokens=8000,
            temperature=0.2
        )

        # Parse + detect skip logic
        requirement_obj = parse_requirement_response(raw_response)

        if requirement_obj is None:
            print(f"[REQ] Chunk {chunk_num}/{total_chunks}: Not a requirement → skip")
            return None

        print(f"[REQ] Chunk {chunk_num}/{total_chunks}: Requirement extracted successfully")
        return requirement_obj

    except Exception as e:
        print(f"[REQ] Chunk {chunk_num}/{total_chunks}: Error: {e}")
        return None


async def generate_testcases_with_retries(requirement_obj, chunk_num, total_chunks, max_tc_retries=2):
    """
    Generate test cases for a requirement object, with retries on validation failure.
    Returns:
        list of valid test cases
        or None (if all retries fail)
    """

    if requirement_obj is None:
        # No requirement → skip
        return None

    # Build final test-case prompt
    prompt = create_chunk_prompt(requirement_obj)

    attempt = 0
    while attempt <= max_tc_retries:
        attempt += 1
        print(f"[TC] Chunk {chunk_num}/{total_chunks}: Attempt {attempt}")

        try:
            # Claude async call
            raw_response = await invoke_bedrock_claude_async(
                prompt=prompt,
                max_tokens=8000,
                temperature=0.3
            )

            # Parse + validate
            tc_obj = parse_testcase_response(raw_response)

            if tc_obj is not None:
                print(f"[TC] Chunk {chunk_num}/{total_chunks}: Test cases generated successfully")
                return tc_obj

            else:
                print(f"[TC] Chunk {chunk_num}/{total_chunks}: Invalid TC response — retrying...")

        except Exception as e:
            print(f"[TC] Chunk {chunk_num}/{total_chunks}: Error: {e}")
            # Retry on exception as well

    # If all retries fail:
    print(f"[TC] Chunk {chunk_num}/{total_chunks}: All retries failed → skipping")
    return None


async def process_chunk_async(chunk_data, max_tc_retries=2):
    """
    Process a single chunk:
      1) Generate requirement
      2) If invalid → skip
      3) If valid → generate test cases with retry
    Returns:
        (chunk_num, tc_list or None)
    """

    chunk, chunk_num, total_chunks = chunk_data

    try:
        # STEP 1: REQUIREMENT GENERATION
        requirement_obj = await generate_requirement_for_chunk(
            chunk,
            chunk_num,
            total_chunks
        )

        # Skip logic
        if requirement_obj is None:
            print(f"[PROCESS] Chunk {chunk_num}/{total_chunks}: No requirement → skip test cases")
            return chunk_num, None

        # STEP 2: TEST CASE GENERATION WITH RETRIES
        tc_list = await generate_testcases_with_retries(
            requirement_obj,
            chunk_num,
            total_chunks,
            max_tc_retries=max_tc_retries
        )

        # If TC generation failed after retries → skip
        if tc_list is None:
            print(f"[PROCESS] Chunk {chunk_num}/{total_chunks}: Test case generation failed after retries")
            return chunk_num, None

        # SUCCESS
        print(f"[PROCESS] Chunk {chunk_num}/{total_chunks}: Completed successfully")
        return chunk_num, tc_list

    except Exception as e:
        print(f"[PROCESS] Chunk {chunk_num}/{total_chunks}: Unexpected error: {e}")
        return chunk_num, None


import asyncio

async def process_chunks_async(chunks, concurrency_limit=3, max_tc_retries=2):
    """
    Process all chunks asynchronously with controlled concurrency.

    Args:
        chunks: list of text chunks
        concurrency_limit: max number of chunks running at once
        max_tc_retries: retries for test case generation

    Returns:
        dict: {chunk_num: tc_list or None}
    """

    total_chunks = len(chunks)
    chunk_data_list = [(chunk, i + 1, total_chunks) for i, chunk in enumerate(chunks)]

    semaphore = asyncio.Semaphore(concurrency_limit)
    results = {}

    async def worker(chunk_data):
        async with semaphore:
            chunk_num, tc_list = await process_chunk_async(
                chunk_data,
                max_tc_retries=max_tc_retries
            )

            # Only store results where TC list exists
            if tc_list is not None:
                results[chunk_num] = tc_list
            else:
                # requirement invalid OR TC retries failed
                results[chunk_num] = None

    # Create async tasks for each chunk
    tasks = [asyncio.create_task(worker(cd)) for cd in chunk_data_list]

    # Await completion of all chunk processing
    await asyncio.gather(*tasks)

    return results


import asyncio

def generate_test_cases_from_content(
        file_path: str,
        output_file: str,
        concurrency_limit: int = 3,
        max_tc_retries: int = 2
    ):
    """
    Top-level synchronous wrapper that:
        1) Reads DOCX
        2) Extracts + flattens + chunks
        3) Asynchronously processes each chunk
        4) Merges valid test cases
        5) Writes to Excel
    """

    print(f"\n[START] Reading and processing document: {file_path}")

    # ------------------ 1) DOCX Extraction ------------------
    dox_json = docx_to_json_with_image_summaries(file_path)

    # ------------------ 2) Recursive Processing ------------------
    processed = recursive_process(dox_json, max_table_rows=5)

    # ------------------ 3) Flattening ------------------
    flattened = flatten_and_merge_top(processed)

    # ------------------ 4) Chunking ------------------
    chunks = chunk_json_structure(flattened, max_chunk_size=2000)
    print(f"[CHUNKS] Total chunks generated: {len(chunks)}")

    # ------------------ 5) Async Processing of All Chunks ------------------
    chunk_results = asyncio.run(
        process_chunks_async(
            chunks,
            concurrency_limit=concurrency_limit,
            max_tc_retries=max_tc_retries
        )
    )

    # ------------------ 6) Ordering ------------------
    ordered_chunk_nums = sorted(chunk_results.keys())
    all_test_cases = [chunk_results[i] for i in ordered_chunk_nums]

    # ------------------ 7) Merging ------------------
    merged_test_cases = merge_test_cases(all_test_cases)
    print(f"[MERGE] Final valid test cases: {len(merged_test_cases)}")

    # ------------------ 8) Excel Writing ------------------
    json_to_excel(output_file, merged_test_cases)
    print(f"[EXPORT] Completed → {output_file}")

    return merged_test_cases


def main():
    # CHANGE THESE ACCORDING TO YOUR FILES
    file_path = "input.docx"           # your source document
    output_file = "output.xlsx"        # final generated Excel file

    try:
        merged_test_cases = generate_test_cases_from_content(
            file_path=file_path,
            output_file=output_file,
            concurrency_limit=3,
            max_tc_retries=2
        )

        print("\n[MAIN] Pipeline completed successfully!")
        print(f"[MAIN] Total valid test cases generated: {len(merged_test_cases)}")
        print(f"[MAIN] Output written to: {output_file}\n")

    except Exception as e:
        print(f"\n[MAIN] ERROR: {e}\n")


# IMPORTANT  — Python entry point
if __name__ == "__main__":
    main()
