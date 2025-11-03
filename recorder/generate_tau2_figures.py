#!/usr/bin/env python3
"""
Generate figures for tau2-bench distillation study.

This script creates visualizations comparing:
1. Single-teacher (GPT-5) distillation with/without rejection sampling
2. Multi-teacher ensemble distillation with/without rejection sampling

Each experiment generates 4 plots (2x2 grid):
- Bar charts: Overall success rates for airline and retail domains
- Heatmaps: Per-task success rates for airline and retail domains
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from collections import defaultdict


def shorten_model_name(model_name, teachers):
    """Extract model variant name after '-yi-' suffix."""
    if model_name in teachers:
        return model_name
    if '-yi-' in model_name:
        return model_name.split('-yi-')[-1]
    return model_name.split('/')[-1]


def parse_teacher_results(jsonl_path):
    """Parse teacher test results from JSONL file."""
    results = {}
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            task_id = str(data['original_task_id'])
            results[task_id] = data['exact_match']
    return results


def parse_aggregated_results(json_path):
    """Parse aggregated results JSON and organize by model and domain."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    model_results = defaultdict(lambda: defaultdict(list))
    
    for task in data['tasks']:
        task_id = task['task_id']
        domain = task['domain']
        
        for result in task['results']:
            model_name = result['model']
            status = result['status']
            success = 1 if status == 'PASSED' else 0
            
            model_results[model_name][domain].append({
                'task_id': task_id,
                'success': success,
                'status': status
            })
    
    return model_results


def compute_success_rates(results_list):
    """Compute overall success rate from list of results."""
    if not results_list:
        return 0.0
    return sum(r['success'] for r in results_list) / len(results_list)


def compute_per_task_success_rates(results_list):
    """Compute per-task success rates."""
    task_results = defaultdict(list)
    for result in results_list:
        task_results[result['task_id']].append(result['success'])
    
    return {task_id: np.mean(successes) for task_id, successes in task_results.items()}


def load_experiment_data(results_path, airline_test_dir, retail_test_dir, teachers):
    """Load all data for one experiment."""
    print(f"Loading results from: {results_path}")
    model_results = parse_aggregated_results(results_path)
    
    print("Loading teacher baselines...")
    teacher_results = {}
    for teacher in teachers:
        teacher_results[teacher] = {
            'airline': parse_teacher_results(f"{airline_test_dir}{teacher}/test.jsonl"),
            'retail': parse_teacher_results(f"{retail_test_dir}{teacher}/test.jsonl")
        }
    
    return model_results, teacher_results


def print_experiment_summary(model_results, teacher_results, teachers, experiment_name):
    """Print summary statistics for one experiment."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {experiment_name}")
    print('='*60)
    
    print("\nTeacher Baselines:")
    for teacher in teachers:
        for domain in ['airline', 'retail']:
            results = teacher_results[teacher][domain]
            success_rate = np.mean(list(results.values()))
            print(f"  {teacher} ({domain}): {success_rate:.1%}")
    
    print("\nStudent Models:")
    for model_name, domains in model_results.items():
        short_name = shorten_model_name(model_name, teachers)
        for domain in ['airline', 'retail']:
            if domain in domains:
                success_rate = compute_success_rates(domains[domain])
                n_tasks = len(set(r['task_id'] for r in domains[domain]))
                print(f"  {short_name} ({domain}): {success_rate:.1%} ({n_tasks} tasks)")


def plot_bar_chart(ax, model_names, model_success, teacher_success, teacher_colors, 
                   domain, color):
    """Create bar chart with teacher reference lines."""
    bars = ax.bar(range(len(model_names)), model_success, 
                   color=color, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=11, fontweight='bold')
    ax.set_ylabel('Success Rate', fontsize=13, fontweight='bold')
    ax.set_title(f'{domain.title()} Domain Performance', 
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add teacher reference lines
    for teacher, rate in teacher_success.items():
        ax.axhline(y=rate, color=teacher_colors[teacher], linestyle='--', 
                   linewidth=2.5, label=f'{teacher}: {rate:.1%}', alpha=0.8)
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
    
    # Add value labels on bars
    for bar, val in zip(bars, model_success):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')


def plot_heatmap(ax, heatmap_data, heatmap_labels, task_ids, domain):
    """Create heatmap of per-task success rates."""
    sns.heatmap(heatmap_data, ax=ax, cmap='RdYlGn', vmin=0, vmax=1,
                cbar_kws={'label': 'Success Rate', 'shrink': 0.8}, 
                yticklabels=heatmap_labels,
                xticklabels=task_ids,
                linewidths=0.5, linecolor='gray', annot=False)
    ax.set_xlabel('Task ID', fontsize=13, fontweight='bold')
    ax.set_ylabel('Model', fontsize=13, fontweight='bold')
    ax.set_title(f'{domain.title()} Per-Task Success Rates', 
                 fontsize=15, fontweight='bold', pad=15)
    ax.tick_params(axis='x', labelsize=9)
    ax.tick_params(axis='y', labelsize=11, labelrotation=0)


def prepare_bar_chart_data(model_results, teacher_results, teachers, domain):
    """Prepare data for bar charts."""
    model_names = []
    model_success = []
    
    for model_name, domains in model_results.items():
        if domain in domains:
            short_name = shorten_model_name(model_name, teachers)
            model_names.append(short_name)
            model_success.append(compute_success_rates(domains[domain]))
    
    teacher_success = {t: np.mean(list(teacher_results[t][domain].values())) 
                       for t in teachers}
    
    return model_names, model_success, teacher_success


def prepare_heatmap_data(model_results, teacher_results, teachers, domain):
    """Prepare data for heatmap."""
    # Collect all task IDs
    task_ids = sorted(set(
        list(teacher_results[teachers[0]][domain].keys()) + 
        [r['task_id'] for domains in model_results.values() 
         if domain in domains for r in domains[domain]]
    ))
    
    heatmap_data = []
    heatmap_labels = []
    
    # Add teacher data
    for teacher in teachers:
        row = [teacher_results[teacher][domain].get(tid, np.nan) 
               for tid in task_ids]
        heatmap_data.append(row)
        heatmap_labels.append(teacher)
    
    # Add model data
    for model_name, domains in model_results.items():
        if domain in domains:
            short_name = shorten_model_name(model_name, teachers)
            task_success = compute_per_task_success_rates(domains[domain])
            row = [task_success.get(tid, np.nan) for tid in task_ids]
            heatmap_data.append(row)
            heatmap_labels.append(short_name)
    
    return heatmap_data, heatmap_labels, task_ids


def create_visualizations(model_results, teacher_results, teachers, teacher_colors, 
                         output_path):
    """Create 2x2 grid of visualizations and save to file."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # Airline bar chart
    model_names, model_success, teacher_success = prepare_bar_chart_data(
        model_results, teacher_results, teachers, 'airline')
    plot_bar_chart(axes[0, 0], model_names, model_success, teacher_success, 
                   teacher_colors, 'airline', color='steelblue')
    
    # Retail bar chart
    model_names, model_success, teacher_success = prepare_bar_chart_data(
        model_results, teacher_results, teachers, 'retail')
    plot_bar_chart(axes[0, 1], model_names, model_success, teacher_success, 
                   teacher_colors, 'retail', color='coral')
    
    # Airline heatmap
    heatmap_data, heatmap_labels, task_ids = prepare_heatmap_data(
        model_results, teacher_results, teachers, 'airline')
    plot_heatmap(axes[1, 0], heatmap_data, heatmap_labels, task_ids, 'airline')
    
    # Retail heatmap
    heatmap_data, heatmap_labels, task_ids = prepare_heatmap_data(
        model_results, teacher_results, teachers, 'retail')
    plot_heatmap(axes[1, 1], heatmap_data, heatmap_labels, task_ids, 'retail')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved visualization to: {output_path}")
    plt.close()


def main():
    """Main execution function."""
    # Configuration
    TEACHERS = ['gpt-5', 'claude-sonnet-4.5', 'gemini-2.5-pro']
    TEACHER_COLORS = {
        'gpt-5': '#e74c3c', 
        'claude-sonnet-4.5': '#9b59b6', 
        'gemini-2.5-pro': '#3498db'
    }
    
    AIRLINE_TEST_DIR = '/mnt/datasets/tau2-bench/recordings/airline/airline-user-gemini-2.5-flash-til20pct/'
    RETAIL_TEST_DIR = '/mnt/datasets/tau2-bench/recordings/retail/retail-user-gemini-2.5-flash-til20pct/'
    
    # Output directory
    output_dir = Path(__file__).parent / 'figures'
    output_dir.mkdir(exist_ok=True)
    
    # Experiment 1: Single-teacher (GPT-5) distillation
    print("\n" + "="*70)
    print("EXPERIMENT 1: Single-Teacher (GPT-5) Distillation")
    print("="*70)
    
    gpt5_results_path = '/mnt/text/workflow/tau2_airline_retail/qwen3-30b-a3b-instruct-2507-base-gemini-2.5-flash-user-airline-retail-yi/0021_aggregate_evaluation_results/aggregated_results.json'
    gpt5_model_results, teacher_results = load_experiment_data(
        gpt5_results_path, AIRLINE_TEST_DIR, RETAIL_TEST_DIR, TEACHERS)
    
    print_experiment_summary(gpt5_model_results, teacher_results, TEACHERS, 
                            "Single-Teacher (GPT-5)")
    
    create_visualizations(gpt5_model_results, teacher_results, TEACHERS, 
                         TEACHER_COLORS, 
                         output_dir / 'gpt5_distillation.png')
    
    # Experiment 2: Multi-teacher ensemble distillation
    print("\n" + "="*70)
    print("EXPERIMENT 2: Multi-Teacher Ensemble Distillation")
    print("="*70)
    
    ensemble_results_path = '/mnt/text/workflow/tau2_ensemble_airline_retail/qwen3-30b-a3b-instruct-2507-base-gemini-2.5-flash-user-airline-retail-ensemble-yi/0027_aggregate_evaluation_results/aggregated_results.json'
    ensemble_model_results, teacher_results = load_experiment_data(
        ensemble_results_path, AIRLINE_TEST_DIR, RETAIL_TEST_DIR, TEACHERS)
    
    print_experiment_summary(ensemble_model_results, teacher_results, TEACHERS,
                            "Multi-Teacher Ensemble")
    
    create_visualizations(ensemble_model_results, teacher_results, TEACHERS, 
                         TEACHER_COLORS,
                         output_dir / 'ensemble_distillation.png')
    
    print("\n" + "="*70)
    print("ALL VISUALIZATIONS COMPLETE")
    print("="*70)
    print(f"\nOutputs saved to: {output_dir}")
    print("  - gpt5_distillation.png")
    print("  - ensemble_distillation.png")


if __name__ == "__main__":
    main()


