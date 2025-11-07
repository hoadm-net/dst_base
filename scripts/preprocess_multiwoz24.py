"""
Script tiền xử lý MultiWOZ 2.4 dataset
"""

import json
import os
from pathlib import Path
from collections import defaultdict, Counter
from tqdm import tqdm


class MultiWOZ24Preprocessor:
    def __init__(self, data_dir="../data/multiwoz24", output_dir="../data/processed"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Domains trong MultiWOZ
        self.domains = ['hotel', 'restaurant', 'attraction', 'train', 'taxi']
        
        # Data containers
        self.data = None
        self.ontology = None
        self.val_list = []
        self.test_list = []
        
    def load_data(self):
        """Load dữ liệu gốc"""
        print("\n" + "=" * 70)
        print("LOADING DATA")
        print("=" * 70)
        
        # Load main data
        data_file = self.data_dir / "data.json"
        print(f"Loading {data_file}...")
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        print(f"✓ Loaded {len(self.data)} dialogues")
        
        # Load ontology
        ontology_file = self.data_dir / "ontology.json"
        print(f"Loading {ontology_file}...")
        with open(ontology_file, 'r', encoding='utf-8') as f:
            self.ontology = json.load(f)
        print(f"✓ Loaded ontology with {len(self.ontology)} domains")
        
        # Load split files (text files with dialogue IDs including .json extension)
        val_file = self.data_dir / "valListFile.json"
        with open(val_file, 'r', encoding='utf-8') as f:
            self.val_list = [line.strip() for line in f if line.strip()]
        print(f"✓ Val set: {len(self.val_list)} dialogues")
        
        test_file = self.data_dir / "testListFile.json"
        with open(test_file, 'r', encoding='utf-8') as f:
            self.test_list = [line.strip() for line in f if line.strip()]
        print(f"✓ Test set: {len(self.test_list)} dialogues")
        
        train_size = len(self.data) - len(self.val_list) - len(self.test_list)
        print(f"✓ Train set: {train_size} dialogues")
    
    def normalize_slot_name(self, domain, slot):
        """Chuẩn hóa tên slot: domain-slot"""
        return f"{domain}-{slot}".lower()
    
    def extract_belief_state(self, metadata):
        """Trích xuất belief state từ metadata"""
        belief_state = {}
        
        if not metadata:
            return belief_state
        
        for domain in self.domains:
            if domain in metadata:
                domain_data = metadata[domain]
                
                # Lấy book và semi slots
                for slot_type in ['book', 'semi']:
                    if slot_type in domain_data:
                        for slot, value in domain_data[slot_type].items():
                            # Bỏ qua các giá trị rỗng hoặc 'not mentioned'
                            if value and value not in ['', 'none', 'not mentioned']:
                                slot_name = self.normalize_slot_name(domain, slot)
                                belief_state[slot_name] = value
        
        return belief_state
    
    def process_dialogue(self, dialogue_id, dialogue):
        """Xử lý một dialogue"""
        processed = {
            'dialogue_id': dialogue_id,
            'domains': [],
            'turns': []
        }
        
        # Extract domains từ goal
        if 'goal' in dialogue:
            processed['domains'] = [d for d in dialogue['goal'].keys() 
                                   if d in self.domains]
        
        # Accumulated belief state
        accumulated_state = {}
        
        # Process turns
        log = dialogue.get('log', [])
        
        for turn_idx in range(0, len(log), 2):
            # User turn
            if turn_idx < len(log):
                user_turn = log[turn_idx]
                
                turn_data = {
                    'turn_id': turn_idx // 2,
                    'speaker': 'user',
                    'utterance': user_turn.get('text', '').strip(),
                    'belief_state': {},
                    'belief_state_delta': {},
                    'system_response': ''
                }
                
                # System turn - chứa metadata với belief state
                if turn_idx + 1 < len(log):
                    system_turn = log[turn_idx + 1]
                    turn_data['system_response'] = system_turn.get('text', '').strip()
                    
                    # Extract belief state từ system turn metadata
                    if 'metadata' in system_turn:
                        current_state = self.extract_belief_state(system_turn['metadata'])
                        
                        # Calculate delta (slots changed in this turn)
                        delta = {}
                        for slot, value in current_state.items():
                            if slot not in accumulated_state or accumulated_state[slot] != value:
                                delta[slot] = value
                        
                        turn_data['belief_state'] = current_state.copy()
                        turn_data['belief_state_delta'] = delta
                        accumulated_state.update(current_state)
                
                processed['turns'].append(turn_data)
        
        return processed
    
    def split_data(self, processed_dialogues):
        """Chia dữ liệu thành train/val/test"""
        train_data = []
        val_data = []
        test_data = []
        
        for dialogue in processed_dialogues:
            dialogue_id = dialogue['dialogue_id']
            
            if dialogue_id in self.test_list:
                test_data.append(dialogue)
            elif dialogue_id in self.val_list:
                val_data.append(dialogue)
            else:
                train_data.append(dialogue)
        
        return train_data, val_data, test_data
    
    def compute_statistics(self, data, split_name):
        """Tính toán thống kê"""
        stats = {
            'split': split_name,
            'num_dialogues': len(data),
            'num_turns': 0,
            'num_tokens_user': 0,
            'num_tokens_system': 0,
            'domain_counts': Counter(),
            'slot_counts': Counter(),
            'slots_per_turn': [],
            'turns_per_dialogue': []
        }
        
        for dialogue in data:
            num_turns = len(dialogue['turns'])
            stats['num_turns'] += num_turns
            stats['turns_per_dialogue'].append(num_turns)
            stats['domain_counts'].update(dialogue['domains'])
            
            for turn in dialogue['turns']:
                # Count tokens
                stats['num_tokens_user'] += len(turn['utterance'].split())
                stats['num_tokens_system'] += len(turn['system_response'].split())
                
                # Count slots
                num_slots = len(turn['belief_state'])
                stats['slots_per_turn'].append(num_slots)
                stats['slot_counts'].update(turn['belief_state'].keys())
        
        # Calculate averages
        if stats['num_dialogues'] > 0:
            stats['avg_turns_per_dialogue'] = stats['num_turns'] / stats['num_dialogues']
        else:
            stats['avg_turns_per_dialogue'] = 0
        
        if stats['num_turns'] > 0:
            stats['avg_slots_per_turn'] = sum(stats['slots_per_turn']) / stats['num_turns']
            stats['avg_tokens_user'] = stats['num_tokens_user'] / stats['num_turns']
            stats['avg_tokens_system'] = stats['num_tokens_system'] / stats['num_turns']
        else:
            stats['avg_slots_per_turn'] = 0
            stats['avg_tokens_user'] = 0
            stats['avg_tokens_system'] = 0
        
        # Convert Counter to dict for JSON
        stats['domain_counts'] = dict(stats['domain_counts'])
        stats['slot_counts'] = dict(stats['slot_counts'])
        
        # Remove large lists (keep only summaries)
        del stats['slots_per_turn']
        del stats['turns_per_dialogue']
        
        return stats
    
    def save_processed_data(self, train_data, val_data, test_data):
        """Lưu dữ liệu đã xử lý"""
        print("\n" + "=" * 70)
        print("SAVING PROCESSED DATA")
        print("=" * 70)
        
        splits = {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
        
        all_stats = {}
        
        for split_name, split_data in splits.items():
            # Save data
            output_file = self.output_dir / f"{split_name}.json"
            print(f"Saving {output_file}...")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(split_data, f, indent=2, ensure_ascii=False)
            print(f"✓ Saved {split_name}.json ({len(split_data)} dialogues)")
            
            # Compute and save statistics
            stats = self.compute_statistics(split_data, split_name)
            all_stats[split_name] = stats
            
            stats_file = self.output_dir / f"{split_name}_stats.json"
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
        
        # Save combined stats
        combined_stats_file = self.output_dir / "dataset_stats.json"
        with open(combined_stats_file, 'w', encoding='utf-8') as f:
            json.dump(all_stats, f, indent=2, ensure_ascii=False)
        
        # Save ontology
        ontology_file = self.output_dir / "ontology.json"
        with open(ontology_file, 'w', encoding='utf-8') as f:
            json.dump(self.ontology, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved ontology.json")
        
        print(f"\n✓ Tất cả dữ liệu đã được lưu tại: {self.output_dir.absolute()}")
        
        return all_stats
    
    def print_summary(self, all_stats):
        """In tóm tắt thống kê"""
        print("\n" + "=" * 70)
        print("DATASET SUMMARY")
        print("=" * 70)
        
        for split_name in ['train', 'val', 'test']:
            stats = all_stats[split_name]
            
            print(f"\n{split_name.upper()}:")
            print(f"  Dialogues:           {stats['num_dialogues']:>6}")
            print(f"  Turns:               {stats['num_turns']:>6}")
            print(f"  Avg turns/dialogue:  {stats['avg_turns_per_dialogue']:>6.2f}")
            print(f"  Avg slots/turn:      {stats['avg_slots_per_turn']:>6.2f}")
            print(f"  Avg tokens (user):   {stats['avg_tokens_user']:>6.2f}")
            print(f"  Avg tokens (system): {stats['avg_tokens_system']:>6.2f}")
            
            print(f"\n  Top 5 domains:")
            for domain, count in sorted(stats['domain_counts'].items(), 
                                       key=lambda x: x[1], reverse=True)[:5]:
                print(f"    {domain:<15} {count:>5}")
            
            print(f"\n  Top 5 slots:")
            for slot, count in sorted(stats['slot_counts'].items(), 
                                     key=lambda x: x[1], reverse=True)[:5]:
                print(f"    {slot:<25} {count:>6}")
    
    def process_all(self):
        """Xử lý toàn bộ pipeline"""
        print("=" * 70)
        print("BẮT ĐẦU TIỀN XỬ LÝ MULTIWOZ 2.4")
        print("=" * 70)
        
        # Load data
        self.load_data()
        
        # Process dialogues
        print("\n" + "=" * 70)
        print("PROCESSING DIALOGUES")
        print("=" * 70)
        
        processed_dialogues = []
        
        for dialogue_id, dialogue in tqdm(self.data.items(), desc="Processing"):
            try:
                processed = self.process_dialogue(dialogue_id, dialogue)
                processed_dialogues.append(processed)
            except Exception as e:
                print(f"\n✗ Error processing {dialogue_id}: {e}")
                continue
        
        print(f"\n✓ Processed {len(processed_dialogues)} dialogues")
        
        # Split data
        print("\nSplitting data...")
        train_data, val_data, test_data = self.split_data(processed_dialogues)
        print(f"✓ Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        # Save
        all_stats = self.save_processed_data(train_data, val_data, test_data)
        
        # Print summary
        self.print_summary(all_stats)
        
        print("\n" + "=" * 70)
        print("✓ TIỀN XỬ LÝ HOÀN TẤT!")
        print("=" * 70)


def main():
    """Main function"""
    preprocessor = MultiWOZ24Preprocessor(
        data_dir="../data/multiwoz24",
        output_dir="../data/processed"
    )
    preprocessor.process_all()
    
    print("\n🎉 Preprocessing thành công! Dữ liệu đã sẵn sàng để training.")


if __name__ == "__main__":
    main()
