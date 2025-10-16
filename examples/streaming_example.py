"""Example of real-time streaming with WebSocket."""

from llmeval import EvalClient
import time


def main():
    client = EvalClient()
    
    print("Starting WebSocket stream...")
    print("Run evaluations from another terminal to see updates.")
    print("Press Ctrl+C to stop.\n")
    
    def on_update(data):
        """Callback for each WebSocket message."""
        timestamp = time.strftime("%H:%M:%S")
        status_emoji = {
            "passed": "✅",
            "failed": "❌",
            "uncertain": "⚠️",
            "error": "🔴",
            "completed": "✔️"
        }.get(data.get("status", ""), "📊")
        
        print(f"[{timestamp}] {status_emoji} Eval {data['id'][:8]}")
        print(f"  Status: {data.get('status', 'N/A')}")
        print(f"  Model: {data.get('model', 'N/A')}")
        print(f"  Verdict: {data.get('verdict', 'N/A')}")
        print(f"  Latency: {data.get('latency_ms', 'N/A')}ms")
        print()
    
    try:
        # This will block and stream updates
        client.stream_evals(on_update)
    except KeyboardInterrupt:
        print("\nStopped streaming.")


if __name__ == "__main__":
    main()
