"""
Metrics đánh giá cho Dialogue State Tracking
"""

from typing import Dict, List, Tuple
from collections import defaultdict


class DSTMetrics:
    """Các metrics chuẩn cho DST evaluation"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset tất cả counters"""
        self.total_turns = 0
        self.correct_joint_goals = 0
        self.correct_slots = 0
        self.total_slots = 0
        
        # Per-slot accuracy
        self.slot_correct = defaultdict(int)
        self.slot_total = defaultdict(int)
        
        # Precision, Recall, F1
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        
        # Turn-level stats
        self.perfect_turns = 0
    
    def update(self, predicted_state: Dict[str, str], 
               ground_truth_state: Dict[str, str]):
        """
        Update metrics với một prediction
        
        Args:
            predicted_state: Dict of slot-value pairs (predicted)
            ground_truth_state: Dict of slot-value pairs (ground truth)
        """
        self.total_turns += 1
        
        # Joint Goal Accuracy - tất cả slots phải đúng
        if predicted_state == ground_truth_state:
            self.correct_joint_goals += 1
            self.perfect_turns += 1
        
        # Slot Accuracy
        all_slots = set(predicted_state.keys()) | set(ground_truth_state.keys())
        
        for slot in all_slots:
            pred_value = predicted_state.get(slot, None)
            true_value = ground_truth_state.get(slot, None)
            
            self.slot_total[slot] += 1
            self.total_slots += 1
            
            if pred_value == true_value:
                self.slot_correct[slot] += 1
                self.correct_slots += 1
        
        # Precision, Recall, F1 calculation
        for slot, value in predicted_state.items():
            if slot in ground_truth_state:
                if value == ground_truth_state[slot]:
                    self.true_positives += 1
                else:
                    self.false_positives += 1
            else:
                self.false_positives += 1
        
        for slot in ground_truth_state:
            if slot not in predicted_state:
                self.false_negatives += 1
    
    def get_joint_goal_accuracy(self) -> float:
        """Joint Goal Accuracy - % turns với tất cả slots đúng"""
        if self.total_turns == 0:
            return 0.0
        return self.correct_joint_goals / self.total_turns
    
    def get_slot_accuracy(self) -> float:
        """Slot Accuracy - % individual slots đúng"""
        if self.total_slots == 0:
            return 0.0
        return self.correct_slots / self.total_slots
    
    def get_precision(self) -> float:
        """Precision = TP / (TP + FP)"""
        denominator = self.true_positives + self.false_positives
        if denominator == 0:
            return 0.0
        return self.true_positives / denominator
    
    def get_recall(self) -> float:
        """Recall = TP / (TP + FN)"""
        denominator = self.true_positives + self.false_negatives
        if denominator == 0:
            return 0.0
        return self.true_positives / denominator
    
    def get_f1_score(self) -> float:
        """F1 Score = 2 * (Precision * Recall) / (Precision + Recall)"""
        precision = self.get_precision()
        recall = self.get_recall()
        
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    def get_per_slot_accuracy(self) -> Dict[str, float]:
        """Accuracy cho từng slot riêng biệt"""
        per_slot = {}
        for slot in self.slot_total:
            if self.slot_total[slot] > 0:
                per_slot[slot] = self.slot_correct[slot] / self.slot_total[slot]
            else:
                per_slot[slot] = 0.0
        return per_slot
    
    def get_summary(self) -> Dict:
        """Lấy tất cả metrics"""
        return {
            'total_turns': self.total_turns,
            'joint_goal_accuracy': self.get_joint_goal_accuracy(),
            'slot_accuracy': self.get_slot_accuracy(),
            'precision': self.get_precision(),
            'recall': self.get_recall(),
            'f1_score': self.get_f1_score(),
            'perfect_turns': self.perfect_turns,
            'perfect_turn_ratio': self.perfect_turns / self.total_turns if self.total_turns > 0 else 0.0
        }
    
    def print_summary(self):
        """In summary ra console"""
        summary = self.get_summary()
        
        print("=" * 70)
        print("DST EVALUATION METRICS")
        print("=" * 70)
        print(f"Total Turns:              {summary['total_turns']:>10}")
        print(f"Perfect Turns:            {summary['perfect_turns']:>10} ({summary['perfect_turn_ratio']:.2%})")
        print("-" * 70)
        print(f"Joint Goal Accuracy:      {summary['joint_goal_accuracy']:>10.2%}")
        print(f"Slot Accuracy:            {summary['slot_accuracy']:>10.2%}")
        print("-" * 70)
        print(f"Precision:                {summary['precision']:>10.2%}")
        print(f"Recall:                   {summary['recall']:>10.2%}")
        print(f"F1 Score:                 {summary['f1_score']:>10.2%}")
        print("=" * 70)
    
    def print_per_slot_accuracy(self, top_k: int = 10):
        """In per-slot accuracy (top-k worst performing slots)"""
        per_slot = self.get_per_slot_accuracy()
        
        # Sort by accuracy (ascending)
        sorted_slots = sorted(per_slot.items(), key=lambda x: x[1])
        
        print("\n" + "=" * 70)
        print(f"PER-SLOT ACCURACY (Top {top_k} worst performing)")
        print("=" * 70)
        print(f"{'Slot':<30} {'Correct':>10} {'Total':>10} {'Accuracy':>10}")
        print("-" * 70)
        
        for slot, accuracy in sorted_slots[:top_k]:
            correct = self.slot_correct[slot]
            total = self.slot_total[slot]
            print(f"{slot:<30} {correct:>10} {total:>10} {accuracy:>10.2%}")
        
        print("=" * 70)


class DSTEvaluator:
    """Evaluator để đánh giá model trên dataset"""
    
    def __init__(self):
        self.metrics = DSTMetrics()
    
    def evaluate_dialogue(self, predicted_turns: List[Dict], 
                         ground_truth_turns: List[Dict]) -> None:
        """
        Evaluate một dialogue
        
        Args:
            predicted_turns: List of predicted turns with belief_state
            ground_truth_turns: List of ground truth turns with belief_state
        """
        # Đảm bảo số turns giống nhau
        min_turns = min(len(predicted_turns), len(ground_truth_turns))
        
        for i in range(min_turns):
            pred_state = predicted_turns[i].get('belief_state', {})
            true_state = ground_truth_turns[i].get('belief_state', {})
            
            self.metrics.update(pred_state, true_state)
    
    def evaluate_dataset(self, predictions: List[Dict], 
                        ground_truth: List[Dict]) -> Dict:
        """
        Evaluate toàn bộ dataset
        
        Args:
            predictions: List of predicted dialogues
            ground_truth: List of ground truth dialogues
            
        Returns:
            Dict of metrics
        """
        self.metrics.reset()
        
        # Create mapping by dialogue_id
        gt_dict = {d['dialogue_id']: d for d in ground_truth}
        
        for pred_dialogue in predictions:
            dialogue_id = pred_dialogue['dialogue_id']
            
            if dialogue_id not in gt_dict:
                continue
            
            gt_dialogue = gt_dict[dialogue_id]
            
            self.evaluate_dialogue(
                pred_dialogue['turns'],
                gt_dialogue['turns']
            )
        
        return self.metrics.get_summary()
    
    def print_results(self, top_k_slots: int = 15):
        """Print evaluation results"""
        self.metrics.print_summary()
        self.metrics.print_per_slot_accuracy(top_k=top_k_slots)
