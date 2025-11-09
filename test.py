"""
Test Script for AI-Based Semantic Evaluation

This script performs basic tests to verify that all components
are working correctly.
"""

import sys
import os
import traceback
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from data_preprocessing import DataPreprocessor
        print("✅ DataPreprocessor imported successfully")
    except Exception as e:
        print(f"❌ Failed to import DataPreprocessor: {e}")
        return False
    
    try:
        from semantic_evaluator import SemanticEvaluator
        print("✅ SemanticEvaluator imported successfully")
    except Exception as e:
        print(f"❌ Failed to import SemanticEvaluator: {e}")
        return False
    
    try:
        from feedback_generator import LocalFeedbackGenerator
        print("✅ LocalFeedbackGenerator imported successfully")
    except Exception as e:
        print(f"❌ Failed to import LocalFeedbackGenerator: {e}")
        return False
    
    try:
        from evaluation_metrics import EvaluationMetrics
        print("✅ EvaluationMetrics imported successfully")
    except Exception as e:
        print(f"❌ Failed to import EvaluationMetrics: {e}")
        return False
    
    return True

def test_data_preprocessing():
    """Test data preprocessing functionality."""
    print("\nTesting data preprocessing...")
    
    try:
        from data_preprocessing import DataPreprocessor
        
        # Test with sample data
        sample_data = {
            'question': ['What is machine learning?'],
            'model_answer': ['Machine learning is a subset of AI.'],
            'student_answer': ['ML is when computers learn from data.'],
            'score': [7.5]
        }
        
        preprocessor = DataPreprocessor()
        
        # Test text cleaning
        cleaned_text = preprocessor.clean_text("  This is a test.  ")
        assert cleaned_text == "This is a test."
        print("✅ Text cleaning works correctly")
        
        # Test data validation
        import pandas as pd
        df = pd.DataFrame(sample_data)
        preprocessor.data = df
        
        is_valid = preprocessor.validate_data()
        assert is_valid == True
        print("✅ Data validation works correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Data preprocessing test failed: {e}")
        traceback.print_exc()
        return False

def test_feedback_generation():
    """Test feedback generation functionality."""
    print("\nTesting feedback generation...")
    
    try:
        from feedback_generator import LocalFeedbackGenerator
        
        generator = LocalFeedbackGenerator()
        
        # Test feedback generation
        feedback = generator.generate_feedback(
            "What is machine learning?",
            "Machine learning is a subset of artificial intelligence.",
            "ML is when computers learn from data.",
            7.5
        )
        
        # Check that feedback has required components
        assert 'overall_feedback' in feedback
        assert 'coverage' in feedback
        assert 'relevance' in feedback
        assert 'grammar' in feedback
        assert 'coherence' in feedback
        
        print("✅ Feedback generation works correctly")
        return True
        
    except Exception as e:
        print(f"❌ Feedback generation test failed: {e}")
        traceback.print_exc()
        return False

def test_evaluation_metrics():
    """Test evaluation metrics functionality."""
    print("\nTesting evaluation metrics...")
    
    try:
        from evaluation_metrics import EvaluationMetrics
        import numpy as np
        
        evaluator = EvaluationMetrics()
        
        # Test with sample data
        true_scores = [7.0, 8.0, 6.0, 9.0, 7.5]
        predicted_scores = [7.2, 7.8, 6.1, 8.9, 7.4]
        
        metrics = evaluator.compute_all_metrics(true_scores, predicted_scores)
        
        # Check that metrics are computed
        assert 'mse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert 'pearson_correlation' in metrics
        
        print("✅ Evaluation metrics work correctly")
        return True
        
    except Exception as e:
        print(f"❌ Evaluation metrics test failed: {e}")
        traceback.print_exc()
        return False

def test_file_structure():
    """Test that required files exist."""
    print("\nTesting file structure...")
    
    required_files = [
        'app.py',
        'demo.py',
        'requirements.txt',
        'README.md',
        'config.py',
        'src/data_preprocessing.py',
        'src/semantic_evaluator.py',
        'src/bert_scorer.py',
        'src/feedback_generator.py',
        'src/evaluation_metrics.py',
        'data/sample_dataset.csv'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files exist")
        return True

def test_sample_dataset():
    """Test that sample dataset can be loaded."""
    print("\nTesting sample dataset...")
    
    try:
        import pandas as pd
        
        dataset_path = Path('data/sample_dataset.csv')
        if not dataset_path.exists():
            print("❌ Sample dataset not found")
            return False
        
        df = pd.read_csv(dataset_path)
        
        # Check required columns
        required_columns = ['question', 'model_answer', 'student_answer', 'score']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ Missing columns in sample dataset: {missing_columns}")
            return False
        
        # Check data quality
        if len(df) == 0:
            print("❌ Sample dataset is empty")
            return False
        
        print(f"✅ Sample dataset loaded successfully ({len(df)} records)")
        return True
        
    except Exception as e:
        print(f"❌ Sample dataset test failed: {e}")
        return False

def run_all_tests():
    """Run all tests."""
    print("🧪 Running AI-Based Semantic Evaluation Tests")
    print("=" * 50)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Imports", test_imports),
        ("Sample Dataset", test_sample_dataset),
        ("Data Preprocessing", test_data_preprocessing),
        ("Feedback Generation", test_feedback_generation),
        ("Evaluation Metrics", test_evaluation_metrics)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}")
        print("-" * 30)
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} test passed")
            else:
                print(f"❌ {test_name} test failed")
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        print("\n🚀 Next steps:")
        print("1. Run: streamlit run app.py")
        print("2. Open your browser to http://localhost:8501")
        print("3. Initialize models in the sidebar")
        print("4. Start evaluating answers!")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        print("\n🔧 Troubleshooting:")
        print("1. Ensure all dependencies are installed: pip install -r requirements.txt")
        print("2. Check that all files are present")
        print("3. Verify Python version (3.8+ required)")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
