# API Documentation

## Overview

This document provides detailed API documentation for the AI-Based Semantic Evaluation system.

## Core Modules

### 1. DataPreprocessor

Handles data loading, cleaning, and preprocessing for evaluation tasks.

#### Methods

**`load_data(file_path: str, file_type: str = 'csv') -> pd.DataFrame`**
- Loads data from CSV or JSON file
- Parameters:
  - `file_path`: Path to the data file
  - `file_type`: Type of file ('csv' or 'json')
- Returns: pandas DataFrame

**`validate_data(required_columns: List[str] = None) -> bool`**
- Validates that loaded data has required columns
- Parameters:
  - `required_columns`: List of required column names
- Returns: True if validation passes

**`preprocess_data(text_columns: List[str] = None, score_range: Tuple[float, float] = (0, 10)) -> pd.DataFrame`**
- Preprocesses the loaded data
- Parameters:
  - `text_columns`: List of column names containing text
  - `score_range`: Tuple of (min_score, max_score)
- Returns: Preprocessed DataFrame

**`get_data_statistics() -> Dict`**
- Returns statistics about the processed data
- Returns: Dictionary containing data statistics

### 2. SemanticEvaluator

Computes semantic similarity between answers using Sentence-BERT embeddings.

#### Methods

**`compute_similarity(model_answer: str, student_answer: str) -> float`**
- Computes cosine similarity between two answers
- Parameters:
  - `model_answer`: The model/teacher answer
  - `student_answer`: The student answer
- Returns: Cosine similarity score (0-1)

**`batch_compute_similarity(model_answers: List[str], student_answers: List[str]) -> List[float]`**
- Computes similarities for multiple answer pairs
- Parameters:
  - `model_answers`: List of model answers
  - `student_answers`: List of student answers
- Returns: List of similarity scores

**`evaluate_dataset(questions: List[str], model_answers: List[str], student_answers: List[str], include_question_context: bool = True) -> Dict`**
- Evaluates semantic similarity for entire dataset
- Parameters:
  - `questions`: List of questions
  - `model_answers`: List of model answers
  - `student_answers`: List of student answers
  - `include_question_context`: Whether to include question in similarity
- Returns: Dictionary with evaluation results

**`get_similarity_interpretation(similarity_score: float) -> str`**
- Provides interpretation of similarity score
- Parameters:
  - `similarity_score`: Cosine similarity score
- Returns: String interpretation

### 3. BERTScorer

Predicts numeric scores using fine-tuned BERT models.

#### Methods

**`predict(question: str, model_answer: str, student_answer: str) -> float`**
- Predicts score for a single answer pair
- Parameters:
  - `question`: The question
  - `model_answer`: The model answer
  - `student_answer`: The student answer
- Returns: Predicted score (0-10)

**`batch_predict(questions: List[str], model_answers: List[str], student_answers: List[str]) -> List[float]`**
- Predicts scores for multiple answer pairs
- Parameters:
  - `questions`: List of questions
  - `model_answers`: List of model answers
  - `student_answers`: List of student answers
- Returns: List of predicted scores

**`train(train_loader: DataLoader, val_loader: DataLoader, epochs: int = 5, learning_rate: float = 2e-5) -> Dict`**
- Trains the BERT scoring model
- Parameters:
  - `train_loader`: Training data loader
  - `val_loader`: Validation data loader
  - `epochs`: Number of training epochs
  - `learning_rate`: Learning rate for optimization
- Returns: Training history dictionary

### 4. FeedbackGenerator

Generates detailed textual feedback on student answers.

#### Methods

**`generate_feedback(question: str, model_answer: str, student_answer: str, score: float) -> Dict`**
- Generates comprehensive feedback
- Parameters:
  - `question`: The question
  - `model_answer`: The model answer
  - `student_answer`: The student answer
  - `score`: The predicted score
- Returns: Dictionary with feedback components

### 5. EvaluationMetrics

Computes comprehensive evaluation metrics.

#### Methods

**`compute_all_metrics(true_scores: List[float], predicted_scores: List[float], similarities: Optional[List[float]] = None) -> Dict`**
- Computes all evaluation metrics
- Parameters:
  - `true_scores`: Ground truth scores
  - `predicted_scores`: Model predicted scores
  - `similarities`: Optional semantic similarity scores
- Returns: Dictionary with all metrics

**`create_visualizations(true_scores: List[float], predicted_scores: List[float], similarities: Optional[List[float]] = None) -> Dict[str, str]`**
- Creates visualization plots
- Parameters:
  - `true_scores`: Ground truth scores
  - `predicted_scores`: Model predicted scores
  - `similarities`: Optional semantic similarity scores
- Returns: Dictionary mapping plot names to file paths

**`generate_report(output_path: str = 'evaluation_report.json') -> None`**
- Generates comprehensive evaluation report
- Parameters:
  - `output_path`: Path to save the report

## Data Formats

### Input Data Format

CSV file with the following columns:
- `question`: The question text
- `model_answer`: The teacher/model answer
- `student_answer`: The student's answer
- `score`: (Optional) Ground truth scores

### Output Data Format

Evaluation results include:
- `similarities`: List of similarity scores
- `statistics`: Statistical summary
- `detailed_results`: Individual result records

Feedback format:
```json
{
  "overall_feedback": "Overall assessment text",
  "coverage": {
    "feedback": "Coverage assessment",
    "score": 0.8
  },
  "relevance": {
    "feedback": "Relevance assessment", 
    "score": 0.7
  },
  "grammar": {
    "feedback": "Grammar assessment",
    "score": 0.9
  },
  "coherence": {
    "feedback": "Coherence assessment",
    "score": 0.6
  },
  "predicted_score": 7.5,
  "generator": "Local Analysis"
}
```

## Error Handling

All methods include comprehensive error handling:
- Input validation
- Model loading errors
- API connection issues
- File I/O errors

Common exceptions:
- `ValueError`: Invalid input parameters
- `FileNotFoundError`: Missing data files
- `RuntimeError`: Model loading/processing errors

## Performance Considerations

- Models are loaded once and cached
- Batch processing for efficiency
- Memory management for large datasets
- GPU acceleration when available

## Examples

### Basic Usage

```python
from src.data_preprocessing import DataPreprocessor
from src.semantic_evaluator import SemanticEvaluator
from src.feedback_generator import LocalFeedbackGenerator

# Load and preprocess data
preprocessor = DataPreprocessor()
df = preprocessor.load_data('data.csv')
processed_df = preprocessor.preprocess_data()

# Initialize evaluators
evaluator = SemanticEvaluator()
feedback_generator = LocalFeedbackGenerator()

# Evaluate single answer
similarity = evaluator.compute_similarity(model_answer, student_answer)
feedback = feedback_generator.generate_feedback(question, model_answer, student_answer, similarity * 10)

# Batch evaluation
results = evaluator.evaluate_dataset(questions, model_answers, student_answers)
```

### Advanced Usage

```python
from src.bert_scorer import BERTScorer
from src.evaluation_metrics import EvaluationMetrics

# Initialize BERT scorer
scorer = BERTScorer('roberta-base')

# Train model
train_loader, val_loader, test_loader = scorer.prepare_data(df)
history = scorer.train(train_loader, val_loader, epochs=5)

# Make predictions
predictions = scorer.batch_predict(questions, model_answers, student_answers)

# Evaluate performance
metrics = EvaluationMetrics()
results = metrics.compute_all_metrics(true_scores, predictions)
plots = metrics.create_visualizations(true_scores, predictions)
metrics.generate_report('evaluation_report.json')
```
