"""
Check logprobs support across various LLM providers via litellm.

This script tests whether different models support returning logprobs by making
two requests per model - one without logprobs and one with logprobs enabled.
Results are saved to JSON files for analysis.

Example commands:
    # Dry run (preview requests without sending)
    python check_logprobs.py
    
    # Test a specific model
    python check_logprobs.py --model "gpt-4o"
    
    # Send live API requests
    python check_logprobs.py --live
    
    # Test specific model with live requests
    python check_logprobs.py --live --model "fireworks_ai/accounts/fireworks/models/deepseek-v3p1-terminus"

The script tests models including:
    - OpenAI models (gpt-4o, gpt-4o-mini)
    - Anthropic Claude models
    - Google Gemini models
    - Fireworks models (deepseek, glm, kimi, qwen)

Results are saved to data/logprobs_checks/ with timestamps.
"""

import json
import time
import argparse
from datetime import datetime, UTC
from pathlib import Path
from typing import Dict, Any

from litellm import completion


TEST_PROMPT = "Say hello."


def serialize_logprobs(logprobs_obj):
    """Convert logprobs object to a JSON-serializable dict."""
    if logprobs_obj is None:
        return None
    try:
        # Try to extract logprob values and top logprobs if present
        result = {}
        if hasattr(logprobs_obj, 'content'):
            # OpenAI-style logprobs with content list
            content_logprobs = []
            for item in logprobs_obj.content:
                if hasattr(item, 'token'):
                    item_dict = {"token": item.token}
                    if hasattr(item, 'logprob'):
                        item_dict["logprob"] = item.logprob
                    if hasattr(item, 'top_logprobs'):
                        top_logprobs = []
                        for top in item.top_logprobs:
                            top_logprobs.append({
                                "token": top.token,
                                "logprob": top.logprob,
                            })
                        item_dict["top_logprobs"] = top_logprobs
                    content_logprobs.append(item_dict)
            result["content"] = content_logprobs
        return result if result else "present"
    except Exception:
        return "present"  # Fallback if we can't serialize


def check_logprobs_support(model: str, prompt: str = TEST_PROMPT, dry_run: bool = True) -> Dict[str, Any]:
    """Check if a model supports logprobs via litellm."""
    print(f"\n{'='*60}")
    print(f"Checking Model: {model} (via litellm)")
    print(f"{'='*60}")

    results = []
    
    # Test without logprobs
    print("\n--- Test 1: Without logprobs ---")
    params_no_logprobs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 10,
    }
    
    print("Request Params:")
    print(json.dumps(params_no_logprobs, indent=2))
    
    if dry_run:
        result_no_logprobs = {"live": False}
        print("[DRY RUN] Set --live to send")
    else:
        try:
            start = time.time()
            response_no = completion(**params_no_logprobs)
            elapsed = (time.time() - start) * 1000
            result_no_logprobs = {
                "live": True,
                "status": 200,
                "elapsed_ms": int(elapsed),
                "response": response_no._hidden_params if hasattr(response_no, '_hidden_params') else None,
                "text": response_no.choices[0].message.content,
            }
        except Exception as e:
            result_no_logprobs = {
                "live": True,
                "status": -1,
                "error": str(e),
            }
        results.append(result_no_logprobs)
        if result_no_logprobs.get("error"):
            print(f"Error: {result_no_logprobs['error']}")

    # Test with logprobs (with top_logprobs)
    print("\n--- Test 2: With logprobs + top_logprobs ---")
    params_with_logprobs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 10,
        "logprobs": True,
        "top_logprobs": 5,
    }
    
    print("Request Params:")
    print(json.dumps(params_with_logprobs, indent=2))
    
    if dry_run:
        result_with_logprobs = {"live": False, "test": "with_top_logprobs"}
        print("[DRY RUN] Set --live to send")
    else:
        try:
            start = time.time()
            response_with = completion(**params_with_logprobs)
            elapsed = (time.time() - start) * 1000
            logprobs_obj = response_with.choices[0].logprobs if hasattr(response_with.choices[0], 'logprobs') else None
            result_with_logprobs = {
                "live": True,
                "status": 200,
                "elapsed_ms": int(elapsed),
                "text": response_with.choices[0].message.content,
                "logprobs": serialize_logprobs(logprobs_obj),
                "test": "with_top_logprobs",
            }
        except Exception as e:
            result_with_logprobs = {
                "live": True,
                "status": -1,
                "error": str(e),
                "test": "with_top_logprobs",
            }
        results.append(result_with_logprobs)
        if result_with_logprobs.get("error"):
            print(f"Error: {result_with_logprobs['error']}")
            # Try without top_logprobs if this fails
            print("\n--- Test 3: With logprobs only (no top_logprobs) ---")
            params_logprobs_only = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 10,
                "logprobs": True,
            }
            print("Request Params:")
            print(json.dumps(params_logprobs_only, indent=2))
            try:
                start = time.time()
                response_only = completion(**params_logprobs_only)
                elapsed = (time.time() - start) * 1000
                logprobs_obj = response_only.choices[0].logprobs if hasattr(response_only.choices[0], 'logprobs') else None
                result_logprobs_only = {
                    "live": True,
                    "status": 200,
                    "elapsed_ms": int(elapsed),
                    "text": response_only.choices[0].message.content,
                    "logprobs": serialize_logprobs(logprobs_obj),
                    "test": "logprobs_only",
                }
                results.append(result_logprobs_only)
                if result_logprobs_only.get("logprobs"):
                    print("✓ logprobs returned!")
                else:
                    print("✗ No logprobs in response")
            except Exception as e2:
                print(f"Error: {e2}")
        elif result_with_logprobs.get("logprobs"):
            print("✓ logprobs returned!")
        else:
            print("✗ No logprobs in response")

    # Determine support - check if any test returned logprobs
    supports_logprobs = False
    if not dry_run and results:
        for result in results:
            if isinstance(result, dict) and result.get("logprobs") is not None:
                supports_logprobs = True
                break
    
    print("\n--- Result ---")
    if dry_run:
        print("[DRY RUN] Set --live to send actual requests")
    else:
        print(f"Supports logprobs: {supports_logprobs}")
    
    return {
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "model": model,
        "supports_logprobs": supports_logprobs,
        "tests": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check logprobs support via litellm")
    parser.add_argument("--live", action="store_true", help="Send live API requests (default: dry run)")
    parser.add_argument("--model", type=str, help="Model to test (default: test a few common ones)")
    args = parser.parse_args()
    
    # Test models - including all the commented fireworks models from the original
    if args.model:
        test_models = [args.model]
    else:
        test_models = [
            "gpt-5",
            "gpt-4o",
            "anthropic/claude-sonnet-4-5-20250929",
            "gemini/gemini-2.5-pro",
            # Fireworks models (from commented lines in original)
            "fireworks_ai/accounts/fireworks/models/deepseek-v3p1-terminus",
            "fireworks_ai/accounts/fireworks/models/glm-4p6",
            "fireworks_ai/accounts/fireworks/models/kimi-k2-instruct-0905",
        ]
    
    results = []
    for model in test_models:
        try:
            result = check_logprobs_support(model, dry_run=not args.live)
            results.append(result)
        except Exception as e:
            print(f"Failed to test {model}: {e}")
            results.append({
                "model": model,
                "error": str(e),
            })
    
    # Save results
    script_dir = Path(__file__).parent
    out_dir = script_dir / "data" / "logprobs_checks"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"logprobs_check_{ts}.json"
    latest_path = out_dir / "latest.json"
    
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"tests": results}, f, ensure_ascii=False, indent=2)
    with latest_path.open("w", encoding="utf-8") as f:
        json.dump({"tests": results}, f, ensure_ascii=False, indent=2)
    
    print(f"\nSaved results to {out_path}")
    print(f"Also wrote {latest_path}")


if __name__ == "__main__":
    main()