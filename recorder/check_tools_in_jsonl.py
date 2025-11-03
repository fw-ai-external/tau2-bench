#!/usr/bin/env python3
"""
Quick script to validate tools in tau2 JSONL training data.

Usage:
    python recorder/check_tools_in_jsonl.py /path/to/train.jsonl
    python recorder/check_tools_in_jsonl.py /path/to/dir/*.jsonl
"""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter


def check_jsonl_file(file_path: Path) -> dict:
    """Check a single JSONL file for tool consistency."""
    stats = {
        "file": str(file_path),
        "total_records": 0,
        "records_with_tools": 0,
        "records_without_tools": 0,
        "unique_tool_counts": Counter(),
        "tool_names": set(),
        "errors": [],
    }
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                stats["total_records"] += 1
                
                try:
                    record = json.loads(line)
                    
                    # Check if tools field exists
                    if "tools" not in record:
                        stats["records_without_tools"] += 1
                        stats["errors"].append(f"Line {line_num}: Missing 'tools' field")
                        continue
                    
                    tools = record["tools"]
                    
                    # Validate tools is a list
                    if not isinstance(tools, list):
                        stats["errors"].append(f"Line {line_num}: 'tools' is not a list")
                        continue
                    
                    stats["records_with_tools"] += 1
                    stats["unique_tool_counts"][len(tools)] += 1
                    
                    # Collect tool names
                    for tool in tools:
                        if isinstance(tool, dict) and "function" in tool:
                            tool_name = tool["function"].get("name")
                            if tool_name:
                                stats["tool_names"].add(tool_name)
                        else:
                            stats["errors"].append(f"Line {line_num}: Invalid tool format")
                
                except json.JSONDecodeError as e:
                    stats["errors"].append(f"Line {line_num}: JSON decode error: {e}")
    
    except FileNotFoundError:
        stats["errors"].append(f"File not found: {file_path}")
    except Exception as e:
        stats["errors"].append(f"Unexpected error: {e}")
    
    return stats


def print_report(stats: dict):
    """Print a concise report."""
    print(f"\n{'='*70}")
    print(f"File: {stats['file']}")
    print(f"{'='*70}")
    print(f"Total records:         {stats['total_records']}")
    print(f"Records with tools:    {stats['records_with_tools']}")
    print(f"Records without tools: {stats['records_without_tools']}")
    
    if stats['unique_tool_counts']:
        print(f"\nTool count distribution:")
        for count, freq in sorted(stats['unique_tool_counts'].items()):
            print(f"  {count} tools: {freq} records")
    
    if stats['tool_names']:
        print(f"\nUnique tools found ({len(stats['tool_names'])}):")
        for name in sorted(stats['tool_names']):
            print(f"  - {name}")
    
    if stats['errors']:
        print(f"\n⚠️  Errors ({len(stats['errors'])}):")
        for error in stats['errors'][:10]:  # Show first 10 errors
            print(f"  {error}")
        if len(stats['errors']) > 10:
            print(f"  ... and {len(stats['errors']) - 10} more errors")
    else:
        print(f"\n✅ No errors found!")


def main():
    parser = argparse.ArgumentParser(
        description="Check tools field in tau2 JSONL files"
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="JSONL files to check"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show only summary across all files"
    )
    
    args = parser.parse_args()
    
    all_stats = []
    for file_pattern in args.files:
        # Handle glob patterns
        file_path = Path(file_pattern)
        if "*" in str(file_pattern):
            files = list(file_path.parent.glob(file_path.name))
        else:
            files = [file_path]
        
        for file in files:
            stats = check_jsonl_file(file)
            all_stats.append(stats)
            
            if not args.summary:
                print_report(stats)
    
    # Print aggregate summary
    if len(all_stats) > 1 or args.summary:
        print(f"\n{'='*70}")
        print("AGGREGATE SUMMARY")
        print(f"{'='*70}")
        print(f"Total files checked: {len(all_stats)}")
        
        total_records = sum(s["total_records"] for s in all_stats)
        total_with_tools = sum(s["records_with_tools"] for s in all_stats)
        total_errors = sum(len(s["errors"]) for s in all_stats)
        
        print(f"Total records:       {total_records}")
        print(f"Records with tools:  {total_with_tools} ({100*total_with_tools/total_records:.1f}%)")
        print(f"Total errors:        {total_errors}")
        
        # Show all unique tool names across files
        all_tools = set()
        for s in all_stats:
            all_tools.update(s["tool_names"])
        
        if all_tools:
            print(f"\nAll unique tools ({len(all_tools)}):")
            for name in sorted(all_tools):
                print(f"  - {name}")
        
        if total_errors == 0:
            print(f"\n✅ All files passed validation!")
        else:
            print(f"\n⚠️  Found {total_errors} errors across all files")


if __name__ == "__main__":
    main()

