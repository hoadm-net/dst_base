"""
Script phân tích sâu training data để cải thiện rules
"""

import json
import sys
from pathlib import Path
from collections import Counter, defaultdict
import re

sys.path.append(str(Path(__file__).parent.parent))


def analyze_domain_patterns(train_data):
    """Phân tích domain patterns từ dialogues"""
    print("=" * 80)
    print("DOMAIN PATTERN ANALYSIS")
    print("=" * 80)
    
    # Domain keywords
    domain_keywords = defaultdict(Counter)
    domain_first_turns = defaultdict(list)
    
    for dialogue in train_data:
        domains = dialogue.get('domains', [])
        if not domains:
            continue
        
        # Analyze first turn to detect domain
        first_turn = dialogue['turns'][0] if dialogue['turns'] else None
        if first_turn:
            utterance = first_turn['utterance'].lower()
            domain_first_turns[tuple(sorted(domains))].append(utterance)
            
            # Count words per domain
            words = utterance.split()
            for domain in domains:
                domain_keywords[domain].update(words)
    
    print("\nTop keywords per domain:")
    for domain in ['hotel', 'restaurant', 'train', 'attraction', 'taxi']:
        print(f"\n{domain.upper()}:")
        top_keywords = domain_keywords[domain].most_common(15)
        for word, count in top_keywords:
            if len(word) > 2 and word not in ['the', 'and', 'for', 'you', 'can', 'would', 'like']:
                print(f"  {word:<20} {count:>5}")
    
    return domain_keywords


def analyze_slot_filling_patterns(train_data):
    """Phân tích patterns của slot filling"""
    from collections import Counter as CounterCls
    
    print("\n" + "=" * 80)
    print("SLOT FILLING PATTERN ANALYSIS")
    print("=" * 80)
    
    # Patterns: utterance context -> slot-value
    slot_patterns = defaultdict(lambda: defaultdict(int))
    slot_context_words = defaultdict(CounterCls)
    
    for dialogue in train_data:
        for turn in dialogue['turns']:
            utterance = turn['utterance'].lower()
            belief_state = turn.get('belief_state', {})
            delta = turn.get('belief_state_delta', {})
            
            # Focus on slots that changed in this turn
            for slot, value in delta.items():
                if not isinstance(value, str):
                    continue
                
                # Extract words around the value
                value_lower = value.lower()
                if value_lower in utterance:
                    # Find position of value
                    pos = utterance.find(value_lower)
                    
                    # Get 3 words before and after
                    before = utterance[:pos].split()[-3:]
                    after = utterance[pos + len(value_lower):].split()[:3]
                    
                    context = ' '.join(before + ['<VALUE>'] + after)
                    slot_patterns[slot][context] += 1
                    
                    # Count words in utterance for this slot
                    slot_context_words[slot].update(utterance.split())
    
    print("\nTop patterns per slot (first 5 slots):")
    for i, (slot, patterns) in enumerate(list(slot_patterns.items())[:5]):
        print(f"\n{slot}:")
        # Convert to Counter for most_common
        from collections import Counter
        pattern_counter = Counter(patterns)
        for pattern, count in list(pattern_counter.most_common(5)):
            print(f"  {pattern:<60} {count:>3}")
    
    return slot_patterns, slot_context_words


def analyze_value_extraction_clues(train_data):
    """Phân tích clues để extract values"""
    print("\n" + "=" * 80)
    print("VALUE EXTRACTION CLUES ANALYSIS")
    print("=" * 80)
    
    # Analyze what comes before/after values
    before_patterns = defaultdict(Counter)
    after_patterns = defaultdict(Counter)
    
    for dialogue in train_data:
        for turn in dialogue['turns']:
            utterance = turn['utterance'].lower()
            delta = turn.get('belief_state_delta', {})
            
            for slot, value in delta.items():
                if not isinstance(value, str) or len(value) < 3:
                    continue
                
                value_lower = value.lower()
                if value_lower not in utterance:
                    continue
                
                # Get words immediately before and after
                pos = utterance.find(value_lower)
                words_before = utterance[:pos].split()
                words_after = utterance[pos + len(value_lower):].split()
                
                if words_before:
                    before_patterns[slot][words_before[-1]] += 1
                if words_after:
                    after_patterns[slot][words_after[0]] += 1
    
    print("\nWords appearing BEFORE values (top 5 slots):")
    for slot in list(before_patterns.keys())[:5]:
        print(f"\n{slot}:")
        for word, count in before_patterns[slot].most_common(10):
            print(f"  {word:<20} {count:>3}")
    
    print("\n\nWords appearing AFTER values (top 5 slots):")
    for slot in list(after_patterns.keys())[:5]:
        print(f"\n{slot}:")
        for word, count in after_patterns[slot].most_common(10):
            print(f"  {word:<20} {count:>3}")
    
    return before_patterns, after_patterns


def analyze_false_positive_causes(train_data):
    """Phân tích nguyên nhân của false positives"""
    print("\n" + "=" * 80)
    print("FALSE POSITIVE ANALYSIS")
    print("=" * 80)
    
    # Count slots that never appear together
    slot_cooccurrence = defaultdict(lambda: defaultdict(int))
    slot_in_dialogues = defaultdict(set)
    
    for dialogue in train_data:
        dialogue_id = dialogue['dialogue_id']
        all_slots = set()
        
        for turn in dialogue['turns']:
            belief_state = turn.get('belief_state', {})
            for slot in belief_state.keys():
                all_slots.add(slot)
                slot_in_dialogues[slot].add(dialogue_id)
        
        # Count co-occurrences
        for slot1 in all_slots:
            for slot2 in all_slots:
                if slot1 != slot2:
                    slot_cooccurrence[slot1][slot2] += 1
    
    print("\nSlot co-occurrence analysis:")
    print("(If two slots rarely appear together, we shouldn't predict both)")
    
    # Find slots that rarely appear together
    total_dialogues = len(train_data)
    rare_cooccurrences = []
    
    for slot1 in list(slot_cooccurrence.keys())[:10]:
        for slot2, count in slot_cooccurrence[slot1].items():
            slot1_freq = len(slot_in_dialogues[slot1])
            slot2_freq = len(slot_in_dialogues[slot2])
            
            # Expected co-occurrence if independent
            expected = (slot1_freq * slot2_freq) / total_dialogues
            
            if count < expected * 0.1 and slot1_freq > 100 and slot2_freq > 100:
                rare_cooccurrences.append((slot1, slot2, count, slot1_freq, slot2_freq))
    
    print("\nSlots that rarely appear together (should filter):")
    for slot1, slot2, cooc, freq1, freq2 in sorted(rare_cooccurrences, key=lambda x: x[2])[:15]:
        print(f"  {slot1:<25} + {slot2:<25}: {cooc:>4} (freq: {freq1:>4}, {freq2:>4})")
    
    return slot_cooccurrence


def analyze_informative_keywords(train_data):
    """Phân tích keywords có tính phân biệt cao"""
    print("\n" + "=" * 80)
    print("INFORMATIVE KEYWORDS ANALYSIS")
    print("=" * 80)
    
    # TF-IDF style: words that appear frequently with a slot but rarely without it
    slot_word_counts = defaultdict(Counter)
    total_word_counts = Counter()
    turns_per_slot = defaultdict(int)
    total_turns = 0
    
    for dialogue in train_data:
        for turn in dialogue['turns']:
            utterance = turn['utterance'].lower()
            words = utterance.split()
            belief_state = turn.get('belief_state', {})
            
            total_turns += 1
            total_word_counts.update(words)
            
            for slot in belief_state.keys():
                turns_per_slot[slot] += 1
                slot_word_counts[slot].update(words)
    
    # Calculate informativeness score
    print("\nMost informative keywords per slot (top 5 slots):")
    for slot in list(slot_word_counts.keys())[:5]:
        print(f"\n{slot}:")
        
        word_scores = []
        for word, count in slot_word_counts[slot].items():
            if len(word) <= 2 or word in ['the', 'and', 'for', 'you', 'can', 'would', 'like', 'want', 'need']:
                continue
            
            # Score: P(word|slot) / P(word|not slot)
            p_word_given_slot = count / turns_per_slot[slot]
            p_word_overall = total_word_counts[word] / total_turns
            
            if p_word_overall > 0:
                score = p_word_given_slot / p_word_overall
                if score > 1.5:  # Appears more often with this slot
                    word_scores.append((word, score, count))
        
        word_scores.sort(key=lambda x: x[1], reverse=True)
        for word, score, count in word_scores[:10]:
            print(f"  {word:<20} score: {score:>6.2f}  count: {count:>4}")
    
    return slot_word_counts


def main():
    """Main analysis"""
    base_dir = Path(__file__).parent.parent
    train_file = base_dir / "data" / "processed" / "train.json"
    
    print("Loading training data...")
    with open(train_file, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    print(f"✓ Loaded {len(train_data)} dialogues\n")
    
    # Run analyses
    domain_keywords = analyze_domain_patterns(train_data)
    slot_patterns, slot_context = analyze_slot_filling_patterns(train_data)
    before_patterns, after_patterns = analyze_value_extraction_clues(train_data)
    slot_cooccurrence = analyze_false_positive_causes(train_data)
    slot_word_counts = analyze_informative_keywords(train_data)
    
    # Save insights
    insights = {
        'domain_keywords': {k: dict(v.most_common(20)) for k, v in domain_keywords.items()},
        'before_patterns': {k: dict(v.most_common(10)) for k, v in before_patterns.items()},
        'after_patterns': {k: dict(v.most_common(10)) for k, v in after_patterns.items()},
    }
    
    output_file = base_dir / "results" / "training_insights.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(insights, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Insights saved to {output_file}")


if __name__ == "__main__":
    main()
