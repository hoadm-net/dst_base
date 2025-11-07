"""
Script để visualize predictions và analyze errors
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))


def view_predictions(predictions_file: str, num_samples: int = 5):
    """View sample predictions"""
    
    with open(predictions_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("=" * 80)
    print("PREDICTION SAMPLES")
    print("=" * 80)
    print(f"\nTotal dialogues: {data['total_dialogues']}")
    print(f"Timestamp: {data['timestamp']}")
    
    if 'metadata' in data and 'metrics' in data['metadata']:
        metrics = data['metadata']['metrics']
        print(f"\nOverall Metrics:")
        print(f"  Joint Goal Accuracy: {metrics['joint_goal_accuracy']:.2%}")
        print(f"  Slot Accuracy:       {metrics['slot_accuracy']:.2%}")
        print(f"  F1 Score:            {metrics['f1_score']:.2%}")
    
    predictions = data['predictions']
    
    for i, dialogue in enumerate(predictions[:num_samples]):
        print("\n" + "=" * 80)
        print(f"DIALOGUE {i+1}: {dialogue['dialogue_id']}")
        print("=" * 80)
        print(f"Domains: {', '.join(dialogue['domains'])}")
        
        # Show first 3 turns
        for turn in dialogue['turns'][:3]:
            print(f"\n[Turn {turn['turn_id']}]")
            print(f"User: {turn['utterance']}")
            
            print(f"\n  Predicted state ({len(turn['predicted_state'])} slots):")
            if turn['predicted_state']:
                for slot, value in list(turn['predicted_state'].items())[:5]:
                    print(f"    {slot:<25} = {value}")
                if len(turn['predicted_state']) > 5:
                    print(f"    ... and {len(turn['predicted_state']) - 5} more")
            else:
                print("    (empty)")
            
            print(f"\n  Ground truth state ({len(turn['ground_truth_state'])} slots):")
            if turn['ground_truth_state']:
                for slot, value in turn['ground_truth_state'].items():
                    print(f"    {slot:<25} = {value}")
            else:
                print("    (empty)")
            
            if turn['errors']:
                print(f"\n  ❌ Errors ({turn['num_errors']}):")
                for error in turn['errors'][:5]:
                    print(f"    {error['slot']:<25} | Type: {error['error_type']:<15} | Pred: {error['predicted']} | GT: {error['ground_truth']}")
                if len(turn['errors']) > 5:
                    print(f"    ... and {len(turn['errors']) - 5} more errors")
            else:
                print(f"\n  ✓ Perfect prediction!")


def analyze_common_errors(error_analysis_file: str):
    """Analyze common errors"""
    
    with open(error_analysis_file, 'r', encoding='utf-8') as f:
        analysis = json.load(f)
    
    print("\n" + "=" * 80)
    print("ERROR ANALYSIS")
    print("=" * 80)
    
    summary = analysis['summary']
    print(f"\nSummary:")
    print(f"  Total Correct: {summary['total_correct']:>10}")
    print(f"  Total Errors:  {summary['total_errors']:>10}")
    print(f"  Accuracy:      {summary['accuracy']:>10.2%}")
    
    print(f"\nError Types:")
    for error_type, count in analysis['error_types'].items():
        print(f"  {error_type:<20} {count:>10}")
    
    print(f"\nTop 15 Error Slots:")
    for slot, count in list(analysis['top_error_slots'].items())[:15]:
        print(f"  {slot:<30} {count:>10}")
    
    print(f"\nTop 15 Value Errors:")
    for error in analysis['top_value_errors'][:15]:
        pred = error['predicted'] if error['predicted'] else '(none)'
        gt = error['ground_truth'] if error['ground_truth'] else '(none)'
        print(f"  {error['slot']:<25} | Pred: {pred:<20} | GT: {gt:<20} | Count: {error['count']}")


def main():
    """Main function"""
    base_dir = Path(__file__).parent.parent
    results_dir = base_dir / "results"
    
    predictions_file = results_dir / "rule_based_predictions.json"
    error_analysis_file = results_dir / "rule_based_error_analysis.json"
    
    if not predictions_file.exists():
        print(f"❌ Predictions file not found: {predictions_file}")
        return
    
    if not error_analysis_file.exists():
        print(f"❌ Error analysis file not found: {error_analysis_file}")
        return
    
    # View predictions
    view_predictions(str(predictions_file), num_samples=3)
    
    # Analyze errors
    analyze_common_errors(str(error_analysis_file))
    
    print("\n" + "=" * 80)
    print("✓ ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
