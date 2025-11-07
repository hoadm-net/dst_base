"""
Utility functions cho MultiWOZ 2.4 dataset
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import Counter


class DataLoader:
    """Load và truy xuất dữ liệu MultiWOZ"""
    
    def __init__(self, data_dir: str = "data/processed"):
        self.data_dir = Path(data_dir)
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.ontology = None
        self.stats = None
    
    def load_split(self, split: str) -> List[Dict]:
        """Load một split cụ thể"""
        filepath = self.data_dir / f"{split}.json"
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
    
    def load_all(self):
        """Load tất cả splits"""
        print("Loading data...")
        self.train_data = self.load_split('train')
        self.val_data = self.load_split('val')
        self.test_data = self.load_split('test')
        
        print(f"✓ Train: {len(self.train_data)} dialogues")
        print(f"✓ Val: {len(self.val_data)} dialogues")
        print(f"✓ Test: {len(self.test_data)} dialogues")
        
        # Load ontology
        ontology_file = self.data_dir / "ontology.json"
        if ontology_file.exists():
            with open(ontology_file, 'r', encoding='utf-8') as f:
                self.ontology = json.load(f)
            print(f"✓ Ontology loaded")
        
        # Load stats
        stats_file = self.data_dir / "dataset_stats.json"
        if stats_file.exists():
            with open(stats_file, 'r', encoding='utf-8') as f:
                self.stats = json.load(f)
            print(f"✓ Statistics loaded")
    
    def get_dialogue(self, dialogue_id: str, split: str = None) -> Optional[Dict]:
        """Lấy một dialogue theo ID"""
        if split:
            data = getattr(self, f"{split}_data")
            if data:
                for dialogue in data:
                    if dialogue['dialogue_id'] == dialogue_id:
                        return dialogue
        else:
            # Search all splits
            for split in ['train', 'val', 'test']:
                dialogue = self.get_dialogue(dialogue_id, split)
                if dialogue:
                    return dialogue
        
        return None
    
    def get_dialogues_by_domain(self, domain: str, split: str = 'train') -> List[Dict]:
        """Lấy các dialogues theo domain"""
        data = getattr(self, f"{split}_data")
        if not data:
            return []
        
        return [d for d in data if domain in d.get('domains', [])]
    
    def get_slot_values(self, slot_name: str) -> List[str]:
        """Lấy tất cả các giá trị của một slot từ ontology"""
        if not self.ontology:
            return []
        
        # Parse slot name (format: domain-slot)
        parts = slot_name.split('-')
        if len(parts) != 2:
            return []
        
        domain, slot = parts
        
        # Look in ontology
        if domain in self.ontology:
            domain_ontology = self.ontology[domain]
            
            # Check in semi slots
            if 'semi' in domain_ontology and slot in domain_ontology['semi']:
                return domain_ontology['semi'][slot]
            
            # Check in book slots
            if 'book' in domain_ontology and slot in domain_ontology['book']:
                return domain_ontology['book'][slot]
        
        return []


class DataAnalyzer:
    """Phân tích và thống kê dữ liệu"""
    
    @staticmethod
    def analyze_dialogue_length(data: List[Dict]) -> Dict:
        """Phân tích độ dài dialogues"""
        lengths = [len(d['turns']) for d in data]
        
        return {
            'min': min(lengths) if lengths else 0,
            'max': max(lengths) if lengths else 0,
            'mean': sum(lengths) / len(lengths) if lengths else 0,
            'total': sum(lengths)
        }
    
    @staticmethod
    def analyze_domain_distribution(data: List[Dict]) -> Dict:
        """Phân tích phân bố domains"""
        single_domain = 0
        multi_domain = 0
        domain_counter = Counter()
        
        for dialogue in data:
            domains = dialogue.get('domains', [])
            num_domains = len(domains)
            
            if num_domains == 1:
                single_domain += 1
            elif num_domains > 1:
                multi_domain += 1
            
            domain_counter.update(domains)
        
        return {
            'single_domain': single_domain,
            'multi_domain': multi_domain,
            'domain_counts': dict(domain_counter)
        }
    
    @staticmethod
    def analyze_slot_distribution(data: List[Dict]) -> Dict:
        """Phân tích phân bố slots"""
        slot_counter = Counter()
        turns_with_slots = 0
        total_turns = 0
        
        for dialogue in data:
            for turn in dialogue['turns']:
                total_turns += 1
                belief_state = turn.get('belief_state', {})
                
                if belief_state:
                    turns_with_slots += 1
                    slot_counter.update(belief_state.keys())
        
        return {
            'total_turns': total_turns,
            'turns_with_slots': turns_with_slots,
            'turns_without_slots': total_turns - turns_with_slots,
            'slot_coverage': turns_with_slots / total_turns if total_turns > 0 else 0,
            'most_common_slots': dict(slot_counter.most_common(10))
        }
    
    @staticmethod
    def print_dialogue(dialogue: Dict, max_turns: int = None):
        """In một dialogue ra màn hình"""
        print("=" * 80)
        print(f"Dialogue ID: {dialogue['dialogue_id']}")
        print(f"Domains: {', '.join(dialogue.get('domains', []))}")
        print(f"Number of turns: {len(dialogue['turns'])}")
        print("=" * 80)
        
        turns = dialogue['turns']
        if max_turns:
            turns = turns[:max_turns]
        
        for turn in turns:
            print(f"\n[Turn {turn['turn_id']}]")
            print(f"User: {turn['utterance']}")
            
            if turn.get('belief_state'):
                print(f"Belief State: {json.dumps(turn['belief_state'], indent=2)}")
            
            if turn.get('belief_state_delta'):
                print(f"Delta: {json.dumps(turn['belief_state_delta'], indent=2)}")
            
            if turn.get('system_response'):
                print(f"System: {turn['system_response']}")
            
            print("-" * 80)


class DataExporter:
    """Export dữ liệu sang các format khác"""
    
    @staticmethod
    def to_csv(data: List[Dict], output_file: str):
        """Export dữ liệu sang CSV format (flat structure)"""
        import csv
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'dialogue_id', 'turn_id', 'speaker', 'utterance',
                'belief_state', 'system_response'
            ])
            
            # Rows
            for dialogue in data:
                for turn in dialogue['turns']:
                    writer.writerow([
                        dialogue['dialogue_id'],
                        turn['turn_id'],
                        turn['speaker'],
                        turn['utterance'],
                        json.dumps(turn.get('belief_state', {})),
                        turn.get('system_response', '')
                    ])
        
        print(f"✓ Exported to {output_file}")
    
    @staticmethod
    def to_text(data: List[Dict], output_file: str):
        """Export dữ liệu sang text format (readable)"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for dialogue in data:
                f.write(f"=== {dialogue['dialogue_id']} ===\n")
                f.write(f"Domains: {', '.join(dialogue.get('domains', []))}\n\n")
                
                for turn in dialogue['turns']:
                    f.write(f"[Turn {turn['turn_id']}]\n")
                    f.write(f"User: {turn['utterance']}\n")
                    
                    if turn.get('belief_state'):
                        f.write(f"Belief: {turn['belief_state']}\n")
                    
                    if turn.get('system_response'):
                        f.write(f"System: {turn['system_response']}\n")
                    
                    f.write("\n")
                
                f.write("\n" + "=" * 80 + "\n\n")
        
        print(f"✓ Exported to {output_file}")


def example_usage():
    """Ví dụ sử dụng"""
    # Load data
    loader = DataLoader(data_dir="data/processed")
    loader.load_all()
    
    # Get a specific dialogue
    dialogue = loader.get_dialogue("MUL0001", split='train')
    if dialogue:
        DataAnalyzer.print_dialogue(dialogue, max_turns=3)
    
    # Analyze train data
    print("\n" + "=" * 80)
    print("TRAIN DATA ANALYSIS")
    print("=" * 80)
    
    length_stats = DataAnalyzer.analyze_dialogue_length(loader.train_data)
    print(f"\nDialogue length: {length_stats}")
    
    domain_stats = DataAnalyzer.analyze_domain_distribution(loader.train_data)
    print(f"\nDomain distribution: {domain_stats}")
    
    slot_stats = DataAnalyzer.analyze_slot_distribution(loader.train_data)
    print(f"\nSlot distribution: {slot_stats}")


if __name__ == "__main__":
    example_usage()
