"""Basic usage examples for llmeval SDK."""

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
    
    result = client.run_eval(
        model="anthropic:claude-sonnet-4",
        prompt="What is the capital of France?",
        expected="Paris",
        judge_model="gemini:gemini-1.5-pro"
    )
    
    print(f"\nModel: {result.model}")
    print(f"Prompt: {result.prompt}")
    print(f"Output: {result.model_output}")
    print(f"Expected: {result.expected}")
    print(f"Judge Verdict: {result.judge_verdict}")
    print(f"Passed: {result.passed}")
    print(f"Latency: {result.latency_ms}ms")
    
    # Run batch evaluations
    print("\n" + "="*60)
    print("Running batch evaluations...")
    print("="*60)
    
    evals = [
        {
            "model": "anthropic:claude-sonnet-4",
            "prompt": "What is 2+2?",
            "expected": "4"
        },
        {
            "model": "gemini:gemini-1.5-flash",
            "prompt": "What is 3+3?",
            "expected": "6"
        }
    ]
    
    batch_result = client.run_batch(evals)
    
    print(f"\nBatch ID: {batch_result.batch_id}")
    print(f"Total: {batch_result.total}")
    print(f"Passed: {batch_result.passed}")
    print(f"Failed: {batch_result.failed}")
    print(f"Pass Rate: {batch_result.pass_rate:.1f}%")
    print(f"Avg Latency: {batch_result.average_model_latency_ms}ms")


if __name__ == "__main__":
    main()
