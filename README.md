# MultiWOZ 2.4 - Dialogue State Tracking

Tài liệu về tập dữ liệu MultiWOZ 2.4 và quy trình xử lý cho bài toán Dialogue State Tracking (DST).

## 🎯 Dialogue State Tracking (DST)

**Định nghĩa**: Dialogue State Tracking là nhiệm vụ theo dõi trạng thái của một cuộc hội thoại qua từng turn, xác định các thông tin (slots) mà người dùng đã cung cấp.

**Input**: User utterance (câu nói của người dùng)  
**Output**: Belief State - tập hợp các slot-value pairs

**Ví dụ**:
```
User: "I need a cheap hotel in the centre"
Belief State: {
    "hotel-pricerange": "cheap",
    "hotel-area": "centre"
}
```

### Ứng dụng
- Task-oriented dialogue systems
- Virtual assistants (booking, reservation, customer service)
- Information retrieval trong multi-turn conversations

## 📋 MultiWOZ 2.4

**MultiWOZ 2.4** là phiên bản mới nhất (2020) của tập dữ liệu Multi-Domain Wizard-of-Oz, được sử dụng rộng rãi cho nghiên cứu về Dialogue State Tracking.

### Đặc điểm chính:
- **10,438 dialogues** multi-domain task-oriented
- **7 domains**: Restaurant, Hotel, Attraction, Taxi, Train, Hospital, Police
- **30+ slots** cần tracking
- **142,954 turns** với belief state annotations
- Hỗ trợ **cross-domain dependencies** (user nói về nhiều domains trong 1 dialogue)

### Phân chia dữ liệu:
- **Training set**: 8,438 dialogues (80.8%)
- **Validation set**: 1,000 dialogues (9.6%)
- **Test set**: 1,000 dialogues (9.6%)

### Cải tiến so với MultiWOZ 2.0-2.3:
- Sửa lỗi annotations (typos, inconsistencies)
- Chuẩn hóa slot values
- Improved user goal consistency
- Better dialogue flow quality

## 🚀 Quick Start

### 1. Setup môi trường

```bash
# Tạo và activate virtual environment
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# hoặc: venv\Scripts\activate  # Windows

# Cài đặt dependencies
pip install -r requirements.txt
```

### 2. Download dataset

```bash
cd scripts
python download_multiwoz24.py
```

Script sẽ:
- Download MULTIWOZ2.4.zip từ GitHub
- Giải nén dữ liệu
- Download ontology và split files
- Verify tính toàn vẹn

### 3. Tiền xử lý dữ liệu

```bash
python preprocess_multiwoz24.py
```

Script sẽ:
- Load dữ liệu gốc
- Chuẩn hóa belief states
- Tính toán delta states (thay đổi giữa các turns)
- Chia thành train/val/test splits
- Tạo statistics

## 📁 Cấu trúc dữ liệu

### Dữ liệu gốc (data/multiwoz24/)
```
data/multiwoz24/
├── data.json              # Dữ liệu dialogues gốc
├── ontology.json          # Định nghĩa domains & slots
├── valListFile.txt        # Danh sách val dialogues
└── testListFile.txt       # Danh sách test dialogues
```

### Dữ liệu đã xử lý (data/processed/)
```
data/processed/
├── train.json             # Training set
├── val.json               # Validation set
├── test.json              # Test set
├── train_stats.json       # Statistics của train set
├── val_stats.json         # Statistics của val set
├── test_stats.json        # Statistics của test set
├── dataset_stats.json     # Combined statistics
└── ontology.json          # Ontology (copy)
```

### Format dữ liệu đã xử lý

Mỗi dialogue được biểu diễn dưới dạng JSON với cấu trúc:

```json
{
  "dialogue_id": "MUL1234",
  "domains": ["hotel", "restaurant"],
  "turns": [
    {
      "turn_id": 0,
      "speaker": "user",
      "utterance": "I need a cheap hotel in the centre",
      "belief_state": {
        "hotel-pricerange": "cheap",
        "hotel-area": "centre"
      },
      "belief_state_delta": {
        "hotel-pricerange": "cheap",
        "hotel-area": "centre"
      },
      "system_response": "I have several options..."
    },
    {
      "turn_id": 1,
      "speaker": "user",
      "utterance": "I also need parking",
      "belief_state": {
        "hotel-pricerange": "cheap",
        "hotel-area": "centre",
        "hotel-parking": "yes"
      },
      "belief_state_delta": {
        "hotel-parking": "yes"
      },
      "system_response": "Sure, let me find hotels with parking..."
    }
  ]
}
```

**Các trường quan trọng**:
- `belief_state`: Cumulative state - tất cả slots từ đầu dialogue đến turn hiện tại
- `belief_state_delta`: Chỉ các slots thay đổi ở turn hiện tại
- `speaker`: "user" hoặc "system"

## 🔄 Quy trình Tiền xử lý

### Bước 1: Download dữ liệu

```bash
python scripts/download_multiwoz24.py
```

**Thực hiện**:
- Download `MULTIWOZ2.4.zip` từ [GitHub repository](https://github.com/smartyfh/MultiWOZ2.4)
- Giải nén vào `data/multiwoz24/`
- Download các file phụ: `ontology.json`, `valListFile.txt`, `testListFile.txt`
- Verify integrity bằng file size và structure

### Bước 2: Preprocessing

```bash
python scripts/preprocess_multiwoz24.py
```

**Các bước xử lý**:

1. **Load raw data**: Đọc `data.json` với 10,438 dialogues

2. **Normalize belief states**:
   - Chuẩn hóa slot names (lowercase, remove spaces)
   - Chuẩn hóa values (lowercase, trim whitespace)
   - Remove slots với value = "none" hoặc ""

3. **Calculate belief state delta**:
   - So sánh belief state của turn hiện tại với turn trước
   - Chỉ giữ lại slots có thay đổi

4. **Split dataset**:
   - Đọc `valListFile.txt` và `testListFile.txt`
   - Chia dữ liệu thành train/val/test
   - Đảm bảo không overlap giữa các splits

5. **Generate statistics**:
   - Số lượng dialogues, turns, tokens
   - Phân bố domains (single-domain vs multi-domain)
   - Phân bố slots (most frequent slots)
   - Average dialogue length

6. **Save processed data**:
   - `train.json`, `val.json`, `test.json`
   - `*_stats.json` cho mỗi split
   - `dataset_stats.json` tổng hợp

### Output Statistics

Sau khi preprocessing, bạn sẽ có:

```json
{
  "num_dialogues": 8438,
  "num_turns": 114034,
  "avg_turns_per_dialogue": 13.5,
  "num_tokens": 1682349,
  "domain_distribution": {
    "single_domain": 3406,
    "multi_domain": 5032
  },
  "top_slots": [
    ["hotel-pricerange", 2453],
    ["restaurant-food", 2376],
    ["train-destination", 2142],
    ...
  ]
}
```

## 📊 Domain & Slot Structure

### Domains và Slots

| Domain | Slots | Ví dụ |
|--------|-------|-------|
| **Restaurant** | food, pricerange, area, name, book_time, book_day, book_people | food=italian, area=centre |
| **Hotel** | pricerange, type, parking, area, stars, internet, name, book_stay, book_day, book_people | type=guesthouse, parking=yes |
| **Train** | departure, destination, day, arriveby, leaveat, book_people | departure=cambridge, destination=london |
| **Taxi** | departure, destination, arriveby, leaveat | departure=hotel, leaveat=10:00 |
| **Attraction** | type, area, name | type=museum, area=centre |

### Slot Value Types

1. **Categorical**: pricerange (cheap/moderate/expensive), area (centre/north/south/east/west)
2. **Open vocabulary**: name (restaurant names, hotel names)
3. **Numeric**: book_people (1-8), book_stay (1-7)
4. **Time**: leaveat, arriveby (HH:MM format)
5. **Date**: day (monday-sunday), book_day (specific dates)

## 🛠️ Utilities & Scripts

### Phân tích dữ liệu

```bash
python scripts/analyze_training_data.py
```

Phân tích:
- Domain-specific keyword patterns
- Slot filling patterns với context
- Value extraction clues
- Slot co-occurrence statistics

### Test dữ liệu

```bash
python scripts/test_data.py
```

Verify:
- Số lượng dialogues và turns
- Format của belief states
- Consistency của annotations

## � Thách thức trong DST

### 1. Multi-domain Conversations
- User chuyển đổi giữa các domains trong cùng dialogue
- Cần theo dõi state của nhiều domains đồng thời

**Ví dụ**:
```
Turn 1: "I need a hotel in the centre" (hotel domain)
Turn 2: "I also want to find a restaurant nearby" (restaurant domain)
Turn 3: "The hotel should be cheap" (back to hotel domain)
```

### 2. Co-reference và Ellipsis
- User dùng đại từ thay vì lặp lại thông tin
- Model cần hiểu ngữ cảnh từ các turns trước

**Ví dụ**:
```
Turn 1: "I want a restaurant in the centre serving italian food"
Turn 2: "What about in the north?" (co-reference to "restaurant serving italian food")
```

### 3. Slot Value Variations
- Cùng một ý nghĩa có nhiều cách diễn đạt
- Cần normalize về canonical form

**Ví dụ**:
```
"cheap" = "inexpensive" = "low price" = "budget-friendly"
"centre" = "center" = "city centre" = "downtown"
```

### 4. Error Propagation
- Lỗi ở turn trước có thể ảnh hưởng các turns sau
- Cumulative belief state cần maintain accuracy

## 🔧 Dependencies

```txt
requests>=2.31.0
tqdm>=4.66.0
pandas>=2.0.0
numpy>=1.24.0
```

Cài đặt:
```bash
pip install -r requirements.txt
```
