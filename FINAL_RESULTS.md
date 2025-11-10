# MultiWOZ 2.4 Dialogue State Tracking - Final Results

## Dataset Verification

✅ **Preprocessing Verification Completed**
- Compared our preprocessing with official MultiWOZ 2.4 repository
- Domain extraction logic matches official implementation
- Active domain detection: correctly identifies domains with actual goal content
- **5 Main Domains**: hotel, restaurant, train, taxi, attraction (as mentioned in the paper)
- **2 Secondary Domains**: hospital (117 dialogues), police (245 dialogues) - exist but rarely used
- **1 Ghost Domain**: bus (exists in ontology with 6 slots but NO dialogues use it)
- **Total**: 6 active domains in practice (7 in ontology, 5 primary + 2 secondary)

## Model Performance

### Rule-Based DST Model Results
- **Joint Goal Accuracy (JGA)**: 36.22%
- **Slot Accuracy**: 25.25%
- **Precision**: 33.30%
- **Recall**: 51.10%
- **F1 Score**: 40.32%

### Benchmark Comparison
| Model | JGA on MultiWOZ 2.4 |
|-------|---------------------|
| MetaASSIST (STAR) | 80.10% |
| ASSIST (STAR) | 79.41% |
| D3ST (XXL) | 75.90% |
| STAR | 73.62% |
| CHAN | 68.25% |
| SOM-DST | 66.78% |
| TripPy | 64.75% |
| **Our Rule-based Model** | **36.22%** |

## Domain Analysis

### Domain Usage in Test Set
**5 Main Domains (as per paper):**
- **train**: 2,266 predictions (ground truth) vs 3,823 (predicted)
- **restaurant**: 2,362 ground truth vs 1,905 predicted  
- **hotel**: 2,396 ground truth vs 4,455 predicted
- **attraction**: 946 ground truth vs 1,144 predicted
- **taxi**: 604 ground truth vs 1,121 predicted

**Secondary Domains:**
- **hospital**: 0 predictions (no test set coverage, but 117 train dialogues)
- **police**: 0 predictions (no test set coverage, but 245 train dialogues)

**Ghost Domain:**
- **bus**: 0 dialogues (exists in ontology but unused)

### Top Performing Slots
1. **restaurant-food**: 80.04%
2. **restaurant-pricerange**: 57.82%
3. **hotel-pricerange**: 55.80%
4. **train-departure**: 54.91%
5. **train-day**: 53.10%

## Error Analysis

- **False Positives**: 7,841 (model predicts slot changes that didn't occur)
- **False Negatives**: 3,967 (model misses actual slot changes)
- **Incorrect Values**: 462 (correct slot but wrong value)
- **Total Errors**: 12,270

## Key Insights

1. **Paper Claims Confirmed**: MultiWOZ 2.4 paper mentions 5 main domains (attraction, hotel, restaurant, taxi, train)
2. **Domain Reality**: 6 domains actually used (5 main + hospital/police), 1 ghost domain (bus in ontology only)
3. **Rule-based Limitations**: 36.22% JGA shows clear limitations compared to neural models (55-80% JGA)
4. **Over-prediction Issue**: Model tends to predict more slots than actual (12,448 vs 8,574 ground truth)
5. **Domain Bias**: Model over-predicts hotel domain and under-predicts restaurant domain
6. **Strong Food Recognition**: Excellent performance on restaurant-food slot (80.04%)
7. **Preprocessing Accuracy**: Our preprocessing correctly handles MultiWOZ 2.4 format and domain structure

## Technical Validation

✅ **Preprocessing matches official MultiWOZ 2.4 repository**
- Domain extraction: `if dom_v and dom_k not in IGNORE_KEYS_IN_GOAL`
- Supports 7 domains as per official ontology
- Correct train/val/test split (8,438/1,000/1,000)
- Proper belief state delta calculation

✅ **Model Architecture**
- Domain detection with keyword matching
- Context-aware value extraction
- Slot co-occurrence filtering
- Confidence thresholds

## Conclusion

This rule-based DST system provides a solid baseline for MultiWOZ 2.4, achieving 36.22% Joint Goal Accuracy. While significantly below state-of-the-art neural models (80%+), it demonstrates the complexity of dialogue state tracking and provides interpretable rules for further analysis.

**Key Findings:**
- **Paper accuracy confirmed**: MultiWOZ 2.4 paper correctly mentions 5 main domains
- **Domain structure**: 5 primary + 2 secondary + 1 ghost domain = 8 ontology domains, 6 active domains
- **Preprocessing validity**: Our pipeline correctly handles the actual domain structure used in practice

The preprocessing pipeline correctly handles MultiWOZ 2.4 format and can serve as a foundation for more sophisticated approaches.