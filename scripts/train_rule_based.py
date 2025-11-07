"""
Train Rule-Based DST Model
Task: Predict belief_state_delta (only slots changed in current turn)
Features:
- Domain detection to filter irrelevant slots
- Context-aware value extraction using patterns
- Slot co-occurrence filtering to reduce false positives
"""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.rule_based import train_improved_rule_based_model, save_rules
from src.evaluation.metrics import DSTEvaluator
from src.evaluation.utils import PredictionSaver


def load_data(filepath):
    """Load processed dialogue data"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def evaluate_model(model, test_data):
    """Evaluate model on test data"""
    from src.evaluation.metrics import DSTMetrics
    
    metrics = DSTMetrics()
    all_predictions = []
    
    print("\nEvaluating on test set...")
    for dialogue in test_data:
        dialogue_history = []
        
        for turn in dialogue['turns']:
            # Only process user turns
            if turn.get('speaker') != 'user':
                continue
                
            # Add user utterance to history
            dialogue_history.append(turn['utterance'])
            
            # Predict belief state delta (only from current utterance)
            predicted_belief = model.predict([turn['utterance']])
            
            # Get ground truth delta (only slots changed in this turn)
            true_belief = {}
            for slot, value in turn.get('belief_state_delta', {}).items():
                if isinstance(value, str) and value != 'none':
                    true_belief[slot] = value
            
            # Update metrics
            metrics.update(predicted_belief, true_belief)
            
            # Store prediction for analysis
            all_predictions.append({
                'dialogue_id': dialogue.get('dialogue_id', 'unknown'),
                'turn_id': turn.get('turn_id', 0),
                'utterance': turn['utterance'],
                'predicted': predicted_belief,
                'ground_truth': true_belief
            })
    
    return metrics, all_predictions


def main():
    # Paths
    data_dir = project_root / 'data' / 'processed'
    results_dir = project_root / 'results'
    results_dir.mkdir(exist_ok=True)
    
    # Load data
    print("Loading data...")
    train_data = load_data(data_dir / 'train.json')
    test_data = load_data(data_dir / 'test.json')
    print(f"✓ Loaded {len(train_data)} training dialogues")
    print(f"✓ Loaded {len(test_data)} test dialogues")
    
    # Train model
    print("\n" + "=" * 80)
    print("TRAINING RULE-BASED DST MODEL")
    print("=" * 80)
    model = train_improved_rule_based_model(train_data)
    
    # Save rules
    save_rules(model.rules, results_dir / 'extracted_rules.json')
    
    # Evaluate on test set
    print("\n" + "=" * 80)
    print("EVALUATION ON TEST SET")
    print("=" * 80)
    metrics_obj, predictions = evaluate_model(model, test_data)
    
    # Get metrics
    metrics = metrics_obj.get_summary()
    
    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Joint Goal Accuracy: {metrics['joint_goal_accuracy']:.2%}")
    print(f"Slot Accuracy: {metrics['slot_accuracy']:.2%}")
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"Recall: {metrics['recall']:.2%}")
    print(f"F1 Score: {metrics['f1_score']:.2%}")
    
    print("\nTop 10 Slot Accuracies:")
    per_slot = metrics_obj.get_per_slot_accuracy()
    sorted_slots = sorted(per_slot.items(), 
                         key=lambda x: x[1], reverse=True)[:10]
    for slot, acc in sorted_slots:
        print(f"  {slot:30s}: {acc:.2%}")
    
    # Save detailed metrics
    metrics_file = results_dir / 'rule_based_metrics.json'
    full_metrics = {
        **metrics,
        'per_slot_accuracy': per_slot
    }
    with open(metrics_file, 'w') as f:
        json.dump(full_metrics, f, indent=2)
    print(f"\n✓ Detailed metrics saved to {metrics_file}")
    
        # Save predictions
    predictions_file = results_dir / 'rule_based_predictions.json'
    with open(predictions_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    print(f"\n✓ Predictions saved to {predictions_file}")
    
    # Error analysis
    print("\n" + "=" * 80)
    print("ERROR ANALYSIS")
    print("=" * 80)
    
    # Compute error stats from predictions
    false_positives = 0
    false_negatives = 0
    incorrect_values = 0
    
    for pred in predictions:
        pred_slots = set(pred['predicted'].keys())
        true_slots = set(pred['ground_truth'].keys())
        
        # False positives: predicted but not in ground truth
        fp = pred_slots - true_slots
        false_positives += len(fp)
        
        # False negatives: in ground truth but not predicted
        fn = true_slots - pred_slots
        false_negatives += len(fn)
        
        # Incorrect values: predicted and in ground truth but wrong value
        for slot in pred_slots & true_slots:
            if pred['predicted'][slot] != pred['ground_truth'][slot]:
                incorrect_values += 1
    
    total_errors = false_positives + false_negatives + incorrect_values
    
    print(f"False Positives: {false_positives:,}")
    print(f"False Negatives: {false_negatives:,}")
    print(f"Incorrect Values: {incorrect_values:,}")
    print(f"Total Errors: {total_errors:,}")
    
    # Save error analysis
    error_stats = {
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'incorrect_values': incorrect_values,
        'total_errors': total_errors,
        'total_turns': metrics['total_turns']
    }
    error_file = results_dir / 'rule_based_error_analysis.json'
    with open(error_file, 'w') as f:
        json.dump(error_stats, f, indent=2)
    print(f"✓ Error analysis saved to {error_file}")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
