<img width="250" height="250" alt="evaluate" src="https://github.com/user-attachments/assets/3071efa3-d512-43ae-8dfb-adfa292c61e0" />

# llmeval - Python SDK for evaluate
## download the evaluate server from https://github.com/RGGH/evaluate

A Python client library for the evaluate LLM evaluation framework.

## Installation

```
pip install llmeval-sdk
```

```bash
pip install -e .
```

For development with all extras:
```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from llmeval import EvalClient

# Initialize the client
client = EvalClient(base_url="http://127.0.0.1:8080")

# Check server health
status = client.health_check()
print(status)

# Get available models
models = client.get_models()
print(f"Available models: {models}")

# Run a single evaluation
result = client.run_eval(
    model="gemini:gemini-2.5-pro",
    prompt="What is the capital of France?",
    expected="Paris",
    judge_model="gemini:gemini-2.5-pro"
)

print(f"Model output: {result.model_output}")
print(f"Judge verdict: {result.judge_verdict}")
print(f"Passed: {result.passed}")
```

### More examples

```python
"""Basic usage examples for llmeval SDK."""

import time
from llmeval.exceptions import APIError, EvalError
from llmeval import EvalClient


def main():
    # Initialize client
    client = EvalClient()
    
    # Check health
    print("Server health:", client.health_check())
    
    # Get available models
    models = client.get_models()
    print(f"\nAvailable models: {models}")
    
    # Run a single evaluation
    print("\n" + "="*60)
    print("Running single evaluation...")
    print("="*60)
    
    try:
        result = client.run_eval(
            model="gemini:gemini-2.5-flash",
            prompt="What is the capital of France?",
            expected="Paris",
            judge_model="gemini:gemini-2.5-flash"
        )
        
        print(f"\nModel: {result.model}")
        print(f"Prompt: {result.prompt}")
        print(f"Output: {result.model_output}")
        print(f"Expected: {result.expected}")
        print(f"Judge Verdict: {result.judge_verdict}")
        print(f"Passed: {result.passed}")
        print(f"Latency: {result.latency_ms}ms")
    except EvalError as e:
        print(f"\nAn evaluation error occurred: {e}")
        print("This may be due to an invalid model name or missing API key on the server.")
    
    # Run batch evaluations
    print("\n" + "="*60)
    print("Running batch evaluations...")
    print("="*60)
    
    try:
        evals = [
            {
                "model": "gemini:gemini-2.5-flash",
                "prompt": "What is 2+2?",
                "expected": "4",
                "judge_model": "gemini:gemini-2.5-flash"
            },
            {
                "model": "gemini:gemini-2.5-flash",
                "prompt": "What is 3+3?",
                "expected": "6",
                "judge_model": "gemini:gemini-2.5-flash"
            }
        ]
        
        initial_batch_result = client.run_batch(evals)
        batch_id = initial_batch_result.batch_id
        print(f"\nBatch evaluation started with ID: {batch_id}")

        
        # The IDs of the individual evals created in the batch
        eval_ids_in_batch = {res.id for res in initial_batch_result.results}
        num_evals = len(eval_ids_in_batch)

        print("Waiting for batch to complete...")

        start_time = time.time()
        timeout = 60  # seconds

        # Poll the history endpoint until all evals in our batch are completed
        completed_results = []
        while True:
            try:
                history = client.get_history().results
                
                # Find results for the evals in our batch that have completed.
                # An eval is complete if it has model_output, or a verdict, or an error.
                completed_results = [
                    r for r in history if r.id in eval_ids_in_batch and 
                    (r.model_output is not None or r.judge_verdict is not None or r.error_message is not None)
                ]

            except APIError as e:
                print(f"\nCould not fetch history: {e}")
                break
            
            print(f"  ... {len(completed_results)}/{num_evals} evals completed.", end="\r")

            if len(completed_results) == num_evals:
                break
            time.sleep(2) 


            if time.time() - start_time > timeout:
                print("\nTimeout waiting for batch to complete!")
                break

        
        print("\n\nBatch evaluation finished!")
        passed_count = sum(1 for r in completed_results if r.judge_verdict == "Pass")
        failed_count = num_evals - passed_count
        pass_rate = (passed_count / num_evals * 100) if num_evals > 0 else 0
        print(f"Total: {num_evals}")
        print(f"Passed: {passed_count}")
        print(f"Failed: {failed_count}")
        print(f"Pass Rate: {pass_rate:.2f}%")
    except EvalError as e:
        print(f"\nAn evaluation error occurred during batch processing: {e}")
    
    # Manage Judge Prompts
    print("\n" + "="*60)
    print("Managing Judge Prompts...")
    print("="*60)

    try:
        # 1. Create a new judge prompt
        print("\nCreating a new judge prompt...")
        new_prompt = client.create_judge_prompt(
            name="Concise Evaluator",
            template="Is the actual output '{{actual}}' the same as '{{expected}}'? Answer with PASS or FAIL.",
            description="A very simple evaluator for exact matches.",
            set_active=True
        )
        print(f"Successfully created and activated prompt version: {new_prompt.version}")

        # 2. Get the active judge prompt
        print("\nFetching active judge prompt...")
        active_prompt = client.get_active_judge_prompt()
        print(f"Active prompt version: {active_prompt.version} (Name: {active_prompt.name})")

        # 3. List all judge prompts
        print("\nListing all judge prompts...")
        all_prompts = client.get_judge_prompts()
        for p in all_prompts:
            print(f"  - Version {p.version}: {p.name} {'(active)' if p.is_active else ''}")

        # 4. Create a second prompt (without setting it active)
        print("\nCreating another judge prompt...")
        other_prompt = client.create_judge_prompt(
            name="Strict Evaluator v2",
            template="Compare:\nExpected: {{expected}}\nActual: {{actual}}\nVerdict: PASS or FAIL",
            description="Requires exact semantic match"
        )
        print(f"Successfully created prompt version: {other_prompt.version}")

        # 5. Set the second prompt as active
        print(f"\nSetting version {other_prompt.version} as active...")
        client.set_active_judge_prompt(version=other_prompt.version)
        print("Successfully set new active version.")
        active_prompt = client.get_active_judge_prompt()
        print(f"New active prompt version: {active_prompt.version} (Name: {active_prompt.name})")

    except APIError as e:
        print(f"\nAn API error occurred: {e}")


if __name__ == "__main__":
    main()


```
## Example output (from the example above)

```bash
❯ uv run --active examples/basic_usage.py
Server health: {'service': 'eval-api', 'status': 'healthy', 'version': '0.1.5'}

Available models: ['anthropic:claude-opus-4', 'anthropic:claude-sonnet-4', 'anthropic:claude-sonnet-4-5', 'anthropic:claude-haiku-4', 'gemini:gemini-2.5-pro', 'gemini:gemini-2.5-flash', 'ollama:llama3', 'ollama:gemma', 'openai:gpt-4o', 'openai:gpt-4o-mini', 'openai:gpt-3.5-turbo']

============================================================
Running single evaluation...
============================================================

Model: gemini:gemini-2.5-flash
Prompt: What is the capital of France?
Output: The capital of France is **Paris**.
Expected: Paris
Judge Verdict: Pass
Passed: True
Latency: 484ms

============================================================
Running batch evaluations...
============================================================

Batch evaluation started with ID: 28bddffc-9940-4127-92d6-d4d8f1ba24b9
Waiting for batch to complete...
  ... 2/2 evals completed.

Batch evaluation finished!
Total: 2
Passed: 2
Failed: 0
Pass Rate: 100.00%

============================================================
Managing Judge Prompts...
============================================================

Creating a new judge prompt...
Successfully created and activated prompt version: 21

Fetching active judge prompt...
Active prompt version: 21 (Name: Concise Evaluator)

Listing all judge prompts...
  - Version 21: Concise Evaluator (active)
  - Version 20: Strict Evaluator v2 
  - Version 19: Concise Evaluator 
  - Version 18: Strict Evaluator v2 
  - Version 17: Concise Evaluator 
  - Version 16: Strict Evaluator v2 
  - Version 15: Concise Evaluator 
  - Version 14: Strict Evaluator v2 
  - Version 13: Concise Evaluator 
  - Version 12: Strict Evaluator v2 
  - Version 11: Concise Evaluator 
  - Version 10: Strict Evaluator v2 
  - Version 9: Concise Evaluator 
  - Version 8: Concise Evaluator 
  - Version 7: Concise Evaluator 
  - Version 6: Concise Evaluator 
  - Version 5: Minimal Prompt 
  - Version 4: Lenient Evaluation 
  - Version 3: Strict Evaluation 
  - Version 2: Test Prompt 
  - Version 1: Default Judge Prompt 

Creating another judge prompt...
Successfully created prompt version: 22

Setting version 22 as active...
Successfully set new active version.
New active prompt version: 22 (Name: Strict Evaluator v2)

```

## Features

- ✅ Simple, intuitive API
- ✅ Type-safe with Pydantic models
- ✅ Batch evaluation support
- ✅ Real-time WebSocket streaming
- ✅ Jupyter notebook integration
- ✅ pandas DataFrame utilities
- ✅ Comprehensive error handling
- ✅ Context manager support

## Documentation

https://github.com/RGGH/llmeval-python-sdk/blob/main/examples/evaluate.ipynb

## Requirements

- Python 3.8+
- requests
- pydantic
- websockets
- pandas

## License

MIT License
