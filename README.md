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
