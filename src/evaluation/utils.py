"""
Evaluation utilities cho DST models
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime


class PredictionSaver:
    """Lưu predictions dưới dạng JSON để debug"""
    
    @staticmethod
    def save_predictions(predictions: List[Dict], 
                        ground_truth: List[Dict],
                        output_path: str,
                        metadata: Dict = None):
        """
        Lưu predictions kèm ground truth để debug
        
        Args:
            predictions: List of predicted dialogues
            ground_truth: List of ground truth dialogues
            output_path: Path to save
            metadata: Additional metadata (model info, metrics, etc.)
        """
        # Create mapping
        gt_dict = {d['dialogue_id']: d for d in ground_truth}
        
        # Compare predictions with ground truth
        results = []
        
        for pred_dialogue in predictions:
            dialogue_id = pred_dialogue['dialogue_id']
            
            if dialogue_id not in gt_dict:
                continue
            
            gt_dialogue = gt_dict[dialogue_id]
            
            # Compare turns
            comparison = {
                'dialogue_id': dialogue_id,
                'domains': pred_dialogue.get('domains', []),
                'turns': []
            }
            
            min_turns = min(len(pred_dialogue['turns']), len(gt_dialogue['turns']))
            
            for i in range(min_turns):
                pred_turn = pred_dialogue['turns'][i]
                gt_turn = gt_dialogue['turns'][i]
                
                pred_state = pred_turn.get('belief_state', {})
                gt_state = gt_turn.get('belief_state', {})
                
                # Check if perfect match
                is_correct = (pred_state == gt_state)
                
                # Find errors
                errors = []
                all_slots = set(pred_state.keys()) | set(gt_state.keys())
                
                for slot in all_slots:
                    pred_val = pred_state.get(slot)
                    gt_val = gt_state.get(slot)
                    
                    if pred_val != gt_val:
                        errors.append({
                            'slot': slot,
                            'predicted': pred_val,
                            'ground_truth': gt_val,
                            'error_type': 'missing' if pred_val is None 
                                         else 'wrong_value' if gt_val is None 
                                         else 'incorrect'
                        })
                
                turn_result = {
                    'turn_id': pred_turn['turn_id'],
                    'utterance': pred_turn['utterance'],
                    'predicted_state': pred_state,
                    'ground_truth_state': gt_state,
                    'is_correct': is_correct,
                    'errors': errors,
                    'num_errors': len(errors)
                }
                
                comparison['turns'].append(turn_result)
            
            results.append(comparison)
        
        # Save to file
        output_data = {
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat(),
            'total_dialogues': len(results),
            'predictions': results
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Predictions saved to {output_path}")
    
    @staticmethod
    def save_error_analysis(predictions: List[Dict],
                           ground_truth: List[Dict],
                           output_path: str):
        """
        Phân tích và lưu các lỗi phổ biến
        
        Args:
            predictions: List of predicted dialogues
            ground_truth: List of ground truth dialogues
            output_path: Path to save
        """
        from collections import Counter
        
        gt_dict = {d['dialogue_id']: d for d in ground_truth}
        
        # Error statistics
        slot_errors = Counter()  # slot -> error count
        error_types = Counter()  # error type -> count
        value_errors = Counter()  # (slot, wrong_value, correct_value) -> count
        
        total_errors = 0
        total_correct = 0
        
        for pred_dialogue in predictions:
            dialogue_id = pred_dialogue['dialogue_id']
            
            if dialogue_id not in gt_dict:
                continue
            
            gt_dialogue = gt_dict[dialogue_id]
            min_turns = min(len(pred_dialogue['turns']), len(gt_dialogue['turns']))
            
            for i in range(min_turns):
                pred_state = pred_dialogue['turns'][i].get('belief_state', {})
                gt_state = gt_dialogue['turns'][i].get('belief_state', {})
                
                all_slots = set(pred_state.keys()) | set(gt_state.keys())
                
                for slot in all_slots:
                    pred_val = pred_state.get(slot)
                    gt_val = gt_state.get(slot)
                    
                    if pred_val != gt_val:
                        slot_errors[slot] += 1
                        total_errors += 1
                        
                        if pred_val is None:
                            error_types['missing'] += 1
                        elif gt_val is None:
                            error_types['false_positive'] += 1
                        else:
                            error_types['wrong_value'] += 1
                            value_errors[(slot, pred_val, gt_val)] += 1
                    else:
                        total_correct += 1
        
        # Prepare analysis
        analysis = {
            'summary': {
                'total_correct': total_correct,
                'total_errors': total_errors,
                'accuracy': total_correct / (total_correct + total_errors) if (total_correct + total_errors) > 0 else 0.0
            },
            'error_types': dict(error_types),
            'top_error_slots': dict(slot_errors.most_common(20)),
            'top_value_errors': [
                {
                    'slot': slot,
                    'predicted': pred,
                    'ground_truth': gt,
                    'count': count
                }
                for (slot, pred, gt), count in value_errors.most_common(30)
            ]
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Error analysis saved to {output_path}")
        
        # Print summary
        print("\n" + "=" * 70)
        print("ERROR ANALYSIS SUMMARY")
        print("=" * 70)
        print(f"Total Correct:     {total_correct:>10}")
        print(f"Total Errors:      {total_errors:>10}")
        print(f"Accuracy:          {analysis['summary']['accuracy']:>10.2%}")
        print("\nError Types:")
        for error_type, count in error_types.most_common():
            print(f"  {error_type:<20} {count:>10}")
        print("\nTop 10 Error Slots:")
        for slot, count in list(slot_errors.most_common(10)):
            print(f"  {slot:<30} {count:>10}")
        print("=" * 70)
