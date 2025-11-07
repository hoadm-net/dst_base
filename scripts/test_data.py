"""
Script test để xem mẫu dữ liệu sau khi tiền xử lý
"""

import sys
sys.path.append('..')

from src.utils import DataLoader, DataAnalyzer
import json


def main():
    print("=" * 80)
    print("KIỂM TRA DỮ LIỆU ĐÃ TIỀN XỬ LÝ")
    print("=" * 80)
    
    # Load data
    loader = DataLoader(data_dir="../data/processed")
    loader.load_all()
    
    print("\n" + "=" * 80)
    print("1. THỐNG KÊ TỔNG QUAN")
    print("=" * 80)
    
    if loader.stats:
        for split in ['train', 'val', 'test']:
            stats = loader.stats[split]
            print(f"\n{split.upper()}:")
            print(f"  Dialogues:           {stats['num_dialogues']:>6}")
            print(f"  Turns:               {stats['num_turns']:>6}")
            print(f"  Avg turns/dialogue:  {stats['avg_turns_per_dialogue']:>6.2f}")
            print(f"  Avg slots/turn:      {stats['avg_slots_per_turn']:>6.2f}")
    
    print("\n" + "=" * 80)
    print("2. MẪU DIALOGUE TỪ TRAIN SET")
    print("=" * 80)
    
    # Lấy dialogue đầu tiên
    sample_dialogue = loader.train_data[0]
    DataAnalyzer.print_dialogue(sample_dialogue, max_turns=3)
    
    print("\n" + "=" * 80)
    print("3. PHÂN TÍCH CHI TIẾT TRAIN SET")
    print("=" * 80)
    
    # Analyze dialogue length
    length_stats = DataAnalyzer.analyze_dialogue_length(loader.train_data)
    print(f"\nĐộ dài dialogue:")
    print(f"  Min:  {length_stats['min']:>3} turns")
    print(f"  Max:  {length_stats['max']:>3} turns")
    print(f"  Mean: {length_stats['mean']:>6.2f} turns")
    
    # Analyze domain distribution
    domain_stats = DataAnalyzer.analyze_domain_distribution(loader.train_data)
    print(f"\nPhân bố domain:")
    print(f"  Single-domain:  {domain_stats['single_domain']:>6}")
    print(f"  Multi-domain:   {domain_stats['multi_domain']:>6}")
    print(f"\n  Top domains:")
    for domain, count in sorted(domain_stats['domain_counts'].items(), 
                               key=lambda x: x[1], reverse=True)[:5]:
        print(f"    {domain:<15} {count:>6}")
    
    # Analyze slot distribution
    slot_stats = DataAnalyzer.analyze_slot_distribution(loader.train_data)
    print(f"\nPhân bố slots:")
    print(f"  Total turns:         {slot_stats['total_turns']:>6}")
    print(f"  Turns with slots:    {slot_stats['turns_with_slots']:>6}")
    print(f"  Turns without slots: {slot_stats['turns_without_slots']:>6}")
    print(f"  Slot coverage:       {slot_stats['slot_coverage']:>6.1%}")
    print(f"\n  Top 10 slots:")
    for slot, count in list(slot_stats['most_common_slots'].items())[:10]:
        print(f"    {slot:<30} {count:>6}")
    
    print("\n" + "=" * 80)
    print("4. KIỂM TRA BELIEF STATE DELTA")
    print("=" * 80)
    
    # Tìm dialogue có nhiều delta
    max_delta_dialogue = None
    max_delta_count = 0
    
    for dialogue in loader.train_data[:100]:
        delta_count = sum(len(turn['belief_state_delta']) for turn in dialogue['turns'])
        if delta_count > max_delta_count:
            max_delta_count = delta_count
            max_delta_dialogue = dialogue
    
    if max_delta_dialogue:
        print(f"\nDialogue có nhiều thay đổi nhất (trong 100 đầu tiên):")
        print(f"ID: {max_delta_dialogue['dialogue_id']}")
        print(f"Total delta slots: {max_delta_count}")
        print(f"\nFirst 2 turns with deltas:")
        
        count = 0
        for turn in max_delta_dialogue['turns']:
            if turn['belief_state_delta'] and count < 2:
                print(f"\n  Turn {turn['turn_id']}:")
                print(f"  User: {turn['utterance'][:80]}...")
                print(f"  Delta: {json.dumps(turn['belief_state_delta'], indent=4)}")
                count += 1
    
    print("\n" + "=" * 80)
    print("✓ KIỂM TRA HOÀN TẤT!")
    print("=" * 80)


if __name__ == "__main__":
    main()
