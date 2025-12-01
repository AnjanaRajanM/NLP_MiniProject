# two_step_tc_async.py

import asyncio
import json
import functools
from typing import Any, Dict, List, Optional, Tuple

# --- import your existing helpers from the other modules ---
# from your_module import (
#     invoke_bedrock_claude,
#     safe_print,
#     clean_llm_response,
#     create_chunk_prompt,
#     prompt_requirements_extraction,
#     requirements_structure,
# )

# For this example I’ll assume these exist in the same file
# or you import them from app_req.py / app_tc.py etc.


# ---------- ASYNC WRAPPER FOR YOUR EXISTING SYNC LLM CALL ----------

async def invoke_bedrock_claude_async(*, prompt: str, max_tokens: int, temperature: float) -> Any:
    """
    Async wrapper around your existing synchronous `invoke_bedrock_claude`.

    This uses `run_in_executor` which internally uses a thread pool,
    but you don’t have to manage threads yourself. From *your* code’s
    perspective it behaves like async non-blocking I/O.
    """def parse_requirement_response(response: Any) -> Optional[Dict[str, Any]]:
    """
    Parse & clean the requirement LLM response.
    Returns:
        - dict with requirement if valid
        - None if the chunk is 'not a requirement'
    """
    json_str = clean_llm_response(response)
    result = json.loads(json_str)

    # TODO: adapt this logic to your actual schema
    # Example checks:
    text = json.dumps(result).lower()
    if "not a requirement" in text:
        return None

    return result

#2
def parse_testcase_response(response: Any) -> Optional[Dict[str, Any]]:
    """
    Parse & validate test case LLM response.
    Returns None if invalid (to trigger retry).
    """
    json_str_tc = clean_llm_response(response)
    result_tc = json.loads(json_str_tc)

    # TODO: add your own validation rules here
    # Example: ensure it has at least one test case
    if not result_tc:
        return None

    return result_tc

    loop = asyncio.get_running_loop()
    func = functools.partial(
        invoke_bedrock_claude,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return await loop.run_in_executor(None, func)


#3
async def generate_requirement_for_chunk(
    chunk: str,
    chunk_num: int,
    total_chunks: int,
) -> Optional[Dict[str, Any]]:
    """
    Call LLM to generate requirement for one chunk.
    Returns requirement dict or None if 'not a requirement' / error.
    """
    safe_print(f"[REQ] Processing chunk {chunk_num}/{total_chunks}")

    prompt = prompt_requirements_extraction.format(
        requirements_structure=requirements_structure,
        chunk=chunk,
        chunk_num=chunk_num,
        total_chunks=total_chunks,
    )

    try:
        response = await invoke_bedrock_claude_async(
            prompt=prompt,
            max_tokens=8000,
            temperature=0.5,
        )
        req_obj = parse_requirement_response(response)
        if req_obj is None:
            safe_print(f"[REQ] Chunk {chunk_num}/{total_chunks} is NOT a requirement – skipping TC.")
        else:
            safe_print(f"[REQ] Successfully processed requirement for chunk {chunk_num}/{total_chunks}")
        return req_obj
    except Exception as e:
        safe_print(f"[REQ] Error in chunk {chunk_num}/{total_chunks}: {e}")
        return None


async def generate_testcases_with_retries(
    requirement_obj: Dict[str, Any],
    chunk_num: int,
    total_chunks: int,
    max_tc_retries: int = 2,
) -> Optional[Dict[str, Any]]:
    """
    Generate test cases for a requirement with up to `max_tc_retries` retries
    if parsing/validation fails.
    """
    prompt_tc = create_chunk_prompt(requirement_obj)

    attempt = 0
    while attempt <= max_tc_retries:
        attempt += 1
        try:
            safe_print(f"[TC] Generating test cases for chunk {chunk_num}/{total_chunks}, attempt {attempt}")

            response = await invoke_bedrock_claude_async(
                prompt=prompt_tc,
                max_tokens=8000,
                temperature=0.5,
            )

            tc_obj = parse_testcase_response(response)
            if tc_obj is not None:
                safe_print(f"[TC] Successfully processed TC for chunk {chunk_num}/{total_chunks}")
                return tc_obj

            safe_print(f"[TC] Invalid / empty TC for chunk {chunk_num}, will retry…")

        except Exception as e:
            safe_print(f"[TC] Error while generating TC for chunk {chunk_num} (attempt {attempt}): {e}")

    safe_print(f"[TC] Giving up on chunk {chunk_num} after {max_tc_retries + 1} attempts.")
    return None

#4
async def process_chunk_async(
    chunk_data: Tuple[str, int, int],
    max_tc_retries: int = 2,
) -> Tuple[int, Optional[Dict[str, Any]]]:
    """
    Async version of process_chunk:
    - Generate requirement
    - If 'not requirement' -> skip TC
    - Else generate TC with retry
    Returns (chunk_num, tc_result or None)
    """
    chunk, chunk_num, total_chunks = chunk_data

    try:
        # 1) Requirement
        requirement_obj = await generate_requirement_for_chunk(
            chunk=chunk,
            chunk_num=chunk_num,
            total_chunks=total_chunks,
        )

        if requirement_obj is None:
            # Not a requirement → skip test cases
            return chunk_num, None

        # 2) Test cases (with retry)
        tc_obj = await generate_testcases_with_retries(
            requirement_obj=requirement_obj,
            chunk_num=chunk_num,
            total_chunks=total_chunks,
            max_tc_retries=max_tc_retries,
        )

        return chunk_num, tc_obj

    except Exception as e:
        safe_print(f"[CHUNK] Fatal error in chunk {chunk_num}/{total_chunks}: {e}")
        return chunk_num, None

#5
async def process_chunks_async(
    chunks: List[str],
    concurrency_limit: int = 3,
    max_tc_retries: int = 2,
) -> Dict[int, Dict[str, Any]]:
    """
    Process all chunks using async concurrency.
    - concurrency_limit: max number of chunks being processed at once
      (similar to max_workers)
    - returns dict: {chunk_num: tc_result}
    """
    total_chunks = len(chunks)
    chunk_data_list = [(chunk, i + 1, total_chunks) for i, chunk in enumerate(chunks)]

    # Use a semaphore to limit concurrent chunk processing
    semaphore = asyncio.Semaphore(concurrency_limit)
    chunk_results: Dict[int, Dict[str, Any]] = {}

    async def worker(chunk_data: Tuple[str, int, int]):
        chunk, chunk_num, total = chunk_data
        async with semaphore:
            chunk_id, tc_result = await process_chunk_async(
                chunk_data,
                max_tc_retries=max_tc_retries,
            )
            if tc_result is not None:
                chunk_results[chunk_id] = tc_result
            else:
                # Either not a requirement or TC failed even after retries
                safe_print(f"[WORKER] No TC stored for chunk {chunk_id}")

    # Create tasks for all chunks
    tasks = [asyncio.create_task(worker(cd)) for cd in chunk_data_list]

    # Wait for all to finish
    await asyncio.gather(*tasks)

    return chunk_results

#6
def main():
    # TODO: however you currently build chunks (from chunking.py etc.)
    # from chunking import get_chunks
    # chunks = get_chunks(...)
    chunks = []  # replace with your real chunk list of strings

    if not chunks:
        safe_print("No chunks to process.")
        return

    # Run the async pipeline
    chunk_results = asyncio.run(
        process_chunks_async(
            chunks,
            concurrency_limit=3,   # similar to your max_workers
            max_tc_retries=2,
        )
    )

    # Here `chunk_results` is {chunk_num: test_case_json}
    # You can write to file, DB, etc.
    safe_print(f"Completed. Got test cases for {len(chunk_results)} chunks.")

    # Example: write everything into a JSON file
    with open("all_testcases.json", "w", encoding="utf-8") as f:
        json.dump(chunk_results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

