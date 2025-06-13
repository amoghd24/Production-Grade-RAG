#!/usr/bin/env python3
"""
Comprehensive test suite for Notion integration.
Runs multiple test scenarios to ensure robust integration.
"""

import os
import asyncio
import time
from pathlib import Path
from typing import List, Dict, Any
import subprocess
import json

def run_test(test_name: str, env_vars: Dict[str, str] = None) -> Dict[str, Any]:
    """Run a single test with specified environment variables."""
    print(f"\n{'='*60}")
    print(f"🧪 Running Test: {test_name}")
    print('='*60)
    
    # Prepare environment
    original_env = dict(os.environ)
    if env_vars:
        os.environ.update(env_vars)
    
    try:
        # Run the test
        start_time = time.time()
        result = subprocess.run(
            ['python', 'test_notion_integration.py'],
            capture_output=True,
            text=True
        )
        duration = time.time() - start_time
        
        # Parse results
        success = result.returncode == 0
        output = result.stdout
        
        return {
            'name': test_name,
            'success': success,
            'duration': duration,
            'output': output,
            'error': result.stderr if not success else None
        }
        
    finally:
        # Restore original environment
        os.environ.clear()
        os.environ.update(original_env)

async def run_all_tests():
    """Run all test scenarios."""
    tests = [
        # Environment Setup Tests
        {
            'name': 'Basic Environment Setup',
            'env_vars': None
        },
        {
            'name': 'Invalid API Key',
            'env_vars': {'NOTION_API_KEY': 'invalid_key'}
        },
        {
            'name': 'Missing API Key',
            'env_vars': {'NOTION_API_KEY': ''}
        },
        
        # Connection Tests
        {
            'name': 'Valid Connection',
            'env_vars': None
        },
        {
            'name': 'Rate Limiting Test',
            'env_vars': None
        },
        
        # Data Collection Tests
        {
            'name': 'Basic Collection',
            'env_vars': None
        },
        {
            'name': 'Collection with Database ID',
            'env_vars': {'NOTION_DATABASE_ID': os.getenv('NOTION_DATABASE_ID', '')}
        },
        
        # Document Processing Tests
        {
            'name': 'Process Single Document',
            'env_vars': None
        },
        
        # MongoDB Storage Tests
        {
            'name': 'Basic Storage',
            'env_vars': None
        }
    ]
    
    results = []
    for test in tests:
        result = run_test(test['name'], test.get('env_vars'))
        results.append(result)
        
        # Add delay for rate limiting tests
        if 'Rate Limiting' in test['name']:
            time.sleep(2)
    
    return results

def print_summary(results: List[Dict[str, Any]]):
    """Print test summary."""
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    total = len(results)
    passed = sum(1 for r in results if r['success'])
    
    print(f"\nTotal Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    
    print("\nDetailed Results:")
    for result in results:
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        print(f"\n{status} {result['name']}")
        print(f"Duration: {result['duration']:.2f}s")
        if not result['success']:
            print(f"Error: {result['error']}")

def save_results(results: List[Dict[str, Any]]):
    """Save test results to file."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"notion_test_results_{timestamp}.json"
    
    # Clean up results for JSON serialization
    clean_results = []
    for r in results:
        clean_r = r.copy()
        clean_r['output'] = str(clean_r['output'])
        clean_r['error'] = str(clean_r['error'])
        clean_results.append(clean_r)
    
    with open(filename, 'w') as f:
        json.dump(clean_results, f, indent=2)
    
    print(f"\n📝 Results saved to {filename}")

async def main():
    """Main test runner."""
    print("🧠 Second Brain AI Assistant - Comprehensive Notion Integration Test")
    
    results = await run_all_tests()
    print_summary(results)
    save_results(results)

if __name__ == "__main__":
    asyncio.run(main()) 