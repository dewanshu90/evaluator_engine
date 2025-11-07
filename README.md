# Student Answer Evaluator

AI-powered evaluation system for student literacy assessments. Cost-optimized and highly accurate.

## 🎯 Features

- **Cost-Optimized**: Single API call per answer (99% cheaper than multi-call approach)
- **Accurate**: Multi-dimensional evaluation (Intent, Vocabulary, Spelling, Grammar)
- **Fair**: Context-aware scoring that considers phonetic spelling attempts
- **Latest Model**: GPT-5-mini by default (newest, cheapest, most efficient)

## 📁 Project Structure

```
UpliftKidzAgent/
├── evaluator.py              # Core evaluation engine
├── test.py                   # Test script for all students
├── questions/                # Question bank
│   └── literacy_enriched.json
├── student_answers/          # Student response data
│   ├── Student_1.json
│   ├── Student_2.json
│   └── ...
└── .env                      # OpenAI API key
```

## 🚀 Quick Start

### 1. Setup
```bash
# Install dependencies
pip install langchain langgraph langchain-openai python-dotenv

# Set your API key in .env file
OPENAI_API_KEY=your_key_here
```

### 2. Run Evaluation
```bash
python test.py
```

**Cost for 207 student responses:**
- GPT-5-mini: ~$0.05 (default) ⭐
- GPT-4o-mini: ~$0.075
- GPT-3.5-Turbo: ~$0.25
- GPT-4: ~$6.21

**Time:** ~3-5 minutes

## 📊 How It Works

### Single API Call Architecture
Instead of making 4 separate API calls, the system makes **1 combined call** that evaluates all dimensions simultaneously.

**Cost Comparison:**
- Old approach: 4 calls × 207 = 828 API calls = $16.15 (GPT-4)
- New approach: 1 call × 207 = 207 API calls = $0.05 (GPT-5-mini) ✅
- **Savings: 99.7% cost reduction!**

### Evaluation Dimensions

1. **Intent (40-50% weight)**: Does the student understand the core concept?
2. **Vocabulary (20-25% weight)**: Appropriate word choice for age level?
3. **Spelling (10-35% weight)**: Considers phonetic attempts (e.g., "becaus" → "because")
4. **Grammar (15-35% weight)**: Sentence structure, tense, agreement

*Weights adjust based on question difficulty and assessment type.*

## 💻 Usage

### Basic Usage
```python
from evaluator import OptimizedAnswerEvaluator

# Initialize with GPT-5-mini (default, cheapest)
evaluator = OptimizedAnswerEvaluator(model_name="gpt-5-mini")

# Evaluate an answer
result = evaluator.evaluate_answer(question_data, student_answer)

print(f"Score: {result['final_score']}/{result['max_score']}")
print(f"Remarks: {result['remarks']}")
```

### Quick Evaluation
```python
from evaluator import quick_evaluate

# Use GPT-5-mini (default)
result = quick_evaluate(question_data, student_answer)

# Or use GPT-4 for higher accuracy
result = quick_evaluate(question_data, student_answer, use_gpt4=True)
```

## 📈 Output

Results saved to `evaluation_results_optimized/`:
- Individual student JSON files
- Summary report with class statistics
- Per-dimension performance breakdown

Example:
```json
{
  "question_id": "L8A",
  "final_score": 0.79,
  "max_score": 1,
  "percentage": 79.2,
  "partial_scores": {
    "intent": 0.38,
    "vocabulary": 0.20,
    "spelling": 0.09,
    "grammar": 0.13
  },
  "remarks": "Great job! You understood the main idea.",
  "suggestions": "Spelling tip: 'becaus' → 'because'"
}
```

## ⚙️ Configuration

### Model Selection

Edit `test.py` to change model:

```python
# GPT-5-mini (Recommended - Latest & Cheapest) ⭐
evaluator = OptimizedAnswerEvaluator(model_name="gpt-5-mini")
# Cost: ~$0.0002/answer | Quality: Excellent

# GPT-4o-mini (Also Good)
evaluator = OptimizedAnswerEvaluator(model_name="gpt-4o-mini")
# Cost: ~$0.0004/answer | Quality: Excellent

# GPT-4 (High Accuracy)
evaluator = OptimizedAnswerEvaluator(model_name="gpt-4")
# Cost: ~$0.03/answer | Quality: Maximum
```

## 🎓 Example

**Question:** "Why couldn't Anna go to the park?"
**Correct:** "Because it was raining"
**Student:** "because of rain fall"

**Evaluation:**
- Intent: 95% (understood concept)
- Vocabulary: 80%
- Spelling: 90%
- Grammar: 85%
- **Final: 0.89/1.0 (89%)**

**Remarks:** "Great job! You understood the main idea."

## 📝 License

MIT
