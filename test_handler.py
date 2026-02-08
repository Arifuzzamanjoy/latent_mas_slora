#!/usr/bin/env python3
"""
Test script for LatentMAS RunPod Serverless Handler.

Usage:
    # Test locally (handler direct call)
    python test_handler.py --mode local
    
    # Test local API server
    python test_handler.py --mode api
    
    # Test deployed RunPod endpoint
    python test_handler.py --mode runpod --endpoint-id YOUR_ID --api-key YOUR_KEY
"""

import json
import argparse
import requests
from pathlib import Path


def test_local_handler():
    """Test the handler by direct function call"""
    print("\n" + "="*60)
    print("Testing LatentMAS Handler (Local Direct Call)")
    print("="*60)
    
    # Import handler
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from handler import handler
    
    test_cases = [
        {
            "name": "Simple question",
            "input": {
                "prompt": "What is 2+2?"
            }
        },
        {
            "name": "Medical question",
            "input": {
                "prompt": "What are common symptoms of diabetes?",
                "max_tokens": 300
            }
        },
        {
            "name": "Code question",
            "input": {
                "prompt": "Write a Python function to reverse a string",
                "temperature": 0.5
            }
        },
        {
            "name": "With external RAG docs",
            "input": {
                "prompt": "What does the document say about Bitcoin?",
                "rag_documents": [
                    "Bitcoin is a decentralized digital currency created in 2009.",
                    "Ethereum is a blockchain platform with smart contracts."
                ]
            }
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n[Test {i}] {test['name']}")
        print(f"Input: {json.dumps(test['input'], indent=2)[:200]}...")
        
        try:
            result = handler({"input": test["input"]})
            
            if "error" in result:
                print(f"❌ Error: {result['error']}")
            else:
                print(f"✅ Success!")
                print(f"   Domain: {result.get('domain', 'N/A')}")
                print(f"   Response: {result.get('response', '')[:200]}...")
        except Exception as e:
            print(f"❌ Exception: {e}")


def test_local_api(endpoint_url="http://localhost:8000/runsync"):
    """Test against local API server"""
    print("\n" + "="*60)
    print(f"Testing against Local API: {endpoint_url}")
    print("="*60)
    
    payload = {
        "input": {
            "prompt": "What is the capital of Japan?",
            "max_tokens": 200
        }
    }
    
    try:
        print(f"\nSending request...")
        response = requests.post(endpoint_url, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        
        print(f"✅ Response received!")
        print(f"Domain: {result.get('domain', 'N/A')}")
        print(f"Response: {result.get('response', '')[:300]}...")
        
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed. Make sure the local server is running:")
        print("   python handler.py --rp_serve_api")
    except Exception as e:
        print(f"❌ Error: {e}")


def test_runpod_endpoint(endpoint_id: str, api_key: str):
    """Test against deployed RunPod endpoint"""
    print("\n" + "="*60)
    print(f"Testing RunPod Endpoint: {endpoint_id}")
    print("="*60)
    
    try:
        import runpod
        runpod.api_key = api_key
        
        endpoint = runpod.Endpoint(endpoint_id)
        
        print("\nSending request (this may take a moment for cold start)...")
        
        result = endpoint.run_sync({
            "prompt": "What is artificial intelligence?",
            "max_tokens": 300
        })
        
        print(f"✅ Response received!")
        print(f"Result: {json.dumps(result, indent=2)[:500]}...")
        
    except ImportError:
        print("❌ runpod package not installed. Install with: pip install runpod")
    except Exception as e:
        print(f"❌ Error: {e}")


def test_with_json_file(json_path: str):
    """Test with a custom JSON input file"""
    print("\n" + "="*60)
    print(f"Testing with JSON file: {json_path}")
    print("="*60)
    
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from handler import handler
    
    try:
        with open(json_path, 'r') as f:
            test_input = json.load(f)
        
        print(f"Input: {json.dumps(test_input, indent=2)[:300]}...")
        
        result = handler(test_input)
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"✅ Success!")
            print(f"Result: {json.dumps(result, indent=2)}")
            
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Test LatentMAS RunPod Handler",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--mode",
        choices=["local", "api", "runpod", "json"],
        default="local",
        help="Test mode (default: local)"
    )
    parser.add_argument(
        "--endpoint-id",
        type=str,
        help="RunPod endpoint ID (for runpod mode)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="RunPod API key (for runpod mode)"
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000/runsync",
        help="Local API URL (for api mode)"
    )
    parser.add_argument(
        "--json-file",
        type=str,
        default="test_input.json",
        help="JSON input file path (for json mode)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "local":
        test_local_handler()
    elif args.mode == "api":
        test_local_api(args.api_url)
    elif args.mode == "runpod":
        if not args.endpoint_id or not args.api_key:
            print("❌ --endpoint-id and --api-key required for runpod mode")
            return
        test_runpod_endpoint(args.endpoint_id, args.api_key)
    elif args.mode == "json":
        test_with_json_file(args.json_file)


if __name__ == "__main__":
    main()
