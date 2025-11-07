# Evaluation Metrics for Dialogue State Tracking

## 📊 Overview

Tài liệu này mô tả các độ đo được sử dụng để đánh giá model Dialogue State Tracking.

## 🎯 Task Definition

**Input**: User utterance trong một turn của dialogue  
**Output**: Belief State - tập hợp các slot-value pairs mô tả trạng thái hiện tại của dialogue

**Ví dụ**:
```
Utterance: "I need a cheap hotel in the centre"
Belief State: {
    "hotel-pricerange": "cheap",
    "hotel-area": "centre"
}
```

## 📐 Metrics

### 1. Joint Goal Accuracy (JGA)

**Định nghĩa**: Tỷ lệ các turns mà model dự đoán **chính xác 100%** tất cả các slots.

**Công thức**:
$$
\text{JGA} = \frac{\text{Số turns dự đoán chính xác hoàn toàn}}{\text{Tổng số turns}}
$$

**Điều kiện để một turn được coi là chính xác**:
- Tất cả slots trong ground truth đều được predict đúng
- Không có slot nào bị dự đoán thừa (false positive)
- Tất cả values đều khớp chính xác

**Ví dụ**:
```
Turn 1:
  Ground truth: {hotel-pricerange: cheap, hotel-area: centre}
  Predicted:    {hotel-pricerange: cheap, hotel-area: centre}
  → ✅ Exact match

Turn 2:
  Ground truth: {restaurant-food: italian}
  Predicted:    {restaurant-food: italian, restaurant-area: centre}
  → ❌ Có slot thừa (restaurant-area)

Turn 3:
  Ground truth: {train-destination: cambridge}
  Predicted:    {train-destination: london}
  → ❌ Value sai

JGA = 1/3 = 33.33%
```

---

### 2. Slot Accuracy

**Định nghĩa**: Tỷ lệ các slot-value pairs được dự đoán đúng trong tất cả các ground truth slots.

**Công thức**:
$$
\text{Slot Accuracy} = \frac{\text{Số slot-value pairs đúng}}{\text{Tổng số slot-value pairs trong ground truth}}
$$

**Đặc điểm**:
- Chỉ tính các slots có trong ground truth
- Mỗi slot-value pair được tính riêng biệt
- Không penalty cho false positives (slots dự đoán thừa)

**Ví dụ**:
```
Turn 1:
  Ground truth: {hotel-pricerange: cheap, hotel-area: centre}
  Predicted:    {hotel-pricerange: cheap}
  → Correct: 1/2

Turn 2:
  Ground truth: {restaurant-food: italian, restaurant-area: centre}
  Predicted:    {restaurant-food: italian, restaurant-area: centre}
  → Correct: 2/2

Turn 3:
  Ground truth: {train-day: monday}
  Predicted:    {train-day: tuesday}
  → Correct: 0/1

Slot Accuracy = (1 + 2 + 0) / (2 + 2 + 1) = 3/5 = 60%
```

---

### 3. Precision

**Định nghĩa**: Trong số các slots được predict, có bao nhiêu % là đúng.

**Công thức**:
$$
\text{Precision} = \frac{TP}{TP + FP}
$$

**Trong đó**:
- **TP (True Positive)**: Slot được predict đúng (cả slot name và value)
- **FP (False Positive)**: Slot được predict nhưng sai value hoặc không có trong ground truth

**Ý nghĩa**: Precision cao → model thận trọng, ít dự đoán sai

**Ví dụ**:
```
Turn 1:
  Ground truth: {hotel-pricerange: cheap}
  Predicted:    {hotel-pricerange: cheap, hotel-area: centre}
  → TP = 1 (hotel-pricerange đúng)
  → FP = 1 (hotel-area không có trong ground truth)

Turn 2:
  Ground truth: {restaurant-food: italian}
  Predicted:    {restaurant-food: chinese}
  → TP = 0
  → FP = 1 (value sai)

Turn 3:
  Ground truth: {train-day: monday, train-destination: cambridge}
  Predicted:    {train-day: monday}
  → TP = 1 (train-day đúng)
  → FP = 0

Precision = (1 + 0 + 1) / (2 + 1 + 1) = 2/4 = 50%
```

---

### 4. Recall

**Định nghĩa**: Trong số các slots cần predict (ground truth), có bao nhiêu % được tìm thấy đúng.

**Công thức**:
$$
\text{Recall} = \frac{TP}{TP + FN}
$$

**Trong đó**:
- **TP (True Positive)**: Slot được predict đúng
- **FN (False Negative)**: Slot có trong ground truth nhưng không được predict hoặc predict sai value

**Ý nghĩa**: Recall cao → model aggressive, tìm được nhiều slots đúng

**Ví dụ**:
```
Turn 1:
  Ground truth: {hotel-pricerange: cheap, hotel-area: centre}
  Predicted:    {hotel-pricerange: cheap}
  → TP = 1 (hotel-pricerange đúng)
  → FN = 1 (hotel-area bị miss)

Turn 2:
  Ground truth: {restaurant-food: italian}
  Predicted:    {restaurant-food: italian}
  → TP = 1
  → FN = 0

Turn 3:
  Ground truth: {train-day: monday, train-destination: cambridge}
  Predicted:    {}
  → TP = 0
  → FN = 2 (cả 2 slots đều bị miss)

Recall = (1 + 1 + 0) / (2 + 1 + 2) = 2/5 = 40%
```

---

### 5. F1 Score

**Định nghĩa**: Trung bình điều hòa (harmonic mean) của Precision và Recall, cân bằng giữa hai metrics.

**Công thức**:
$$
F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

**Ý nghĩa**: 
- F1 Score cân bằng giữa Precision và Recall
- Hữu ích khi cần đánh giá tổng thể performance
- F1 cao khi cả Precision và Recall đều cao

**Ví dụ**:
```
Precision = 50% = 0.5
Recall = 40% = 0.4

F1 = 2 × (0.5 × 0.4) / (0.5 + 0.4)
   = 2 × 0.2 / 0.9
   = 0.4 / 0.9
   = 0.444
   = 44.44%
```

---

### 6. Per-Slot Accuracy

**Định nghĩa**: Accuracy riêng cho từng slot cụ thể, giúp xác định slot nào model predict tốt/kém.

**Công thức**:
$$
\text{Accuracy}_{\text{slot}} = \frac{\text{Số lần predict đúng slot}}{\text{Tổng số lần slot xuất hiện trong ground truth}}
$$

**Ý nghĩa**: 
- Phân tích chi tiết performance của từng slot
- Xác định điểm mạnh/yếu của model
- Hữu ích cho việc cải thiện targeted

**Ví dụ**:
```
Qua 3 turns, slot "hotel-pricerange" xuất hiện 3 lần:

Turn 1:
  Ground truth: hotel-pricerange = cheap
  Predicted:    hotel-pricerange = cheap
  → ✅ Correct

Turn 3:
  Ground truth: hotel-pricerange = cheap
  Predicted:    hotel-pricerange = expensive
  → ❌ Wrong

Turn 5:
  Ground truth: hotel-pricerange = moderate
  Predicted:    hotel-pricerange = moderate
  → ✅ Correct

Accuracy(hotel-pricerange) = 2/3 = 66.67%
```

---

## 🔍 Confusion Matrix

Phân loại các trường hợp prediction:

### 1. True Positive (TP)
**Định nghĩa**: Slot được predict đúng cả tên và value

**Ví dụ**:
```
Ground truth: {hotel-pricerange: cheap}
Predicted:    {hotel-pricerange: cheap}
→ TP = 1
```

### 2. False Positive (FP)
**Định nghĩa**: Slot được predict nhưng sai hoặc không cần thiết

**Case 1 - Predict slot không có trong ground truth**:
```
Ground truth: {}
Predicted:    {hotel-pricerange: cheap}
→ FP = 1
```

**Case 2 - Predict đúng slot nhưng sai value**:
```
Ground truth: {hotel-pricerange: cheap}
Predicted:    {hotel-pricerange: expensive}
→ FP = 1
```

### 3. False Negative (FN)
**Định nghĩa**: Slot có trong ground truth nhưng không được predict hoặc predict sai

**Ví dụ**:
```
Ground truth: {hotel-pricerange: cheap, hotel-area: centre}
Predicted:    {hotel-pricerange: cheap}
→ FN = 1 (hotel-area bị miss)
```

### 4. Incorrect Value
**Định nghĩa**: Predict đúng slot name nhưng sai value (là một dạng đặc biệt của FP)

**Ví dụ**:
```
Ground truth: {train-destination: cambridge}
Predicted:    {train-destination: london}
→ Incorrect Value = 1
```

---

## 📈 Interpretation Guide

### Joint Goal Accuracy (JGA)
| Mức độ | Giá trị | Ý nghĩa |
|--------|---------|---------|
| **High** | > 50% | Model rất chính xác, phần lớn turns đều predict đúng hoàn toàn |
| **Medium** | 20-50% | Model khá tốt nhưng còn nhiều lỗi nhỏ cần khắc phục |
| **Low** | < 20% | Model có vấn đề nghiêm trọng, cần cải thiện đáng kể |

### Precision vs Recall Trade-off

| Pattern | Đặc điểm | Ý nghĩa |
|---------|----------|---------|
| **High Precision, Low Recall** | Precision > 70%, Recall < 40% | Model **thận trọng**, chỉ predict khi chắc chắn → Ít sai nhưng bỏ sót nhiều |
| **Low Precision, High Recall** | Precision < 40%, Recall > 70% | Model **aggressive**, predict nhiều → Tìm được nhiều nhưng hay sai |
| **Balanced** | Cả hai trong khoảng 50-70% | Model **cân bằng**, là lý tưởng nhất |

### F1 Score Benchmarks

| Loại Model | F1 Score mong đợi |
|------------|-------------------|
| Rule-based | 35-45% |
| Classical ML | 45-55% |
| Neural (LSTM/GRU) | 50-65% |
| Transformer-based | 60-75% |
| State-of-the-art | > 75% |

---

