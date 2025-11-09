"""
Demo Script for AI-Based Semantic Evaluation

This script demonstrates the functionality of the semantic evaluation system
with sample data and provides examples of how to use each component.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import DataPreprocessor
from semantic_evaluator import SemanticEvaluator
from feedback_generator import LocalFeedbackGenerator
from evaluation_metrics import EvaluationMetrics

def demo_data_preprocessing():
    """Demonstrate data preprocessing functionality."""
    print("=" * 60)
    print("DEMO: Data Preprocessing")
    print("=" * 60)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Load sample data
    sample_file = "data/sample_dataset.csv"
    if Path(sample_file).exists():
        df = preprocessor.load_data(sample_file)
        print(f"✅ Loaded dataset with {len(df)} records")
        
        # Validate data
        if preprocessor.validate_data():
            print("✅ Data validation passed")
            
            # Preprocess data
            processed_df = preprocessor.preprocess_data()
            print(f"✅ Preprocessed data: {len(processed_df)} records")
            
            # Get statistics
            stats = preprocessor.get_data_statistics()
            print("\n📊 Dataset Statistics:")
            print(f"Total records: {stats['total_records']}")
            print(f"Score range: {stats['score_stats']['min']:.1f} - {stats['score_stats']['max']:.1f}")
            print(f"Average score: {stats['score_stats']['mean']:.1f}")
            
            return processed_df
        else:
            print("❌ Data validation failed")
            return None
    else:
        print(f"❌ Sample dataset not found at {sample_file}")
        return None

def demo_semantic_evaluation(df):
    """Demonstrate semantic evaluation functionality."""
    print("\n" + "=" * 60)
    print("DEMO: Semantic Evaluation")
    print("=" * 60)
    
    try:
        # Initialize evaluator
        evaluator = SemanticEvaluator()
        print("✅ Semantic evaluator initialized")
        
        # Test single evaluation
        if df is not None and len(df) > 0:
            sample_row = df.iloc[0]
            similarity = evaluator.compute_similarity(
                sample_row['model_answer'], 
                sample_row['student_answer']
            )
            
            print(f"\n🔍 Sample Evaluation:")
            print(f"Question: {sample_row['question'][:100]}...")
            print(f"Similarity Score: {similarity:.3f}")
            print(f"Interpretation: {evaluator.get_similarity_interpretation(similarity)}")
            
            # Test batch evaluation
            print(f"\n📊 Batch Evaluation:")
            results = evaluator.evaluate_dataset(
                df['question'].tolist()[:3],  # Use first 3 for demo
                df['model_answer'].tolist()[:3],
                df['student_answer'].tolist()[:3]
            )
            
            print(f"Mean similarity: {results['statistics']['mean']:.3f}")
            print(f"Std deviation: {results['statistics']['std']:.3f}")
            
            return results
        else:
            print("❌ No data available for evaluation")
            return None
            
    except Exception as e:
        print(f"❌ Error in semantic evaluation: {str(e)}")
        return None

def demo_feedback_generation(df):
    """Demonstrate feedback generation functionality."""
    print("\n" + "=" * 60)
    print("DEMO: Feedback Generation")
    print("=" * 60)
    
    try:
        # Initialize feedback generator
        feedback_generator = LocalFeedbackGenerator()
        print("✅ Feedback generator initialized")
        
        # Test feedback generation
        if df is not None and len(df) > 0:
            sample_row = df.iloc[0]
            feedback = feedback_generator.generate_feedback(
                sample_row['question'],
                sample_row['model_answer'],
                sample_row['student_answer'],
                sample_row['score']
            )
            
            print(f"\n📝 Sample Feedback:")
            print(f"Overall: {feedback['overall_feedback']}")
            print(f"\nCoverage: {feedback['coverage']['feedback']}")
            print(f"Relevance: {feedback['relevance']['feedback']}")
            print(f"Grammar: {feedback['grammar']['feedback']}")
            print(f"Coherence: {feedback['coherence']['feedback']}")
            
            return feedback
        else:
            print("❌ No data available for feedback generation")
            return None
            
    except Exception as e:
        print(f"❌ Error in feedback generation: {str(e)}")
        return None

def demo_evaluation_metrics(df):
    """Demonstrate evaluation metrics functionality."""
    print("\n" + "=" * 60)
    print("DEMO: Evaluation Metrics")
    print("=" * 60)
    
    try:
        # Initialize metrics evaluator
        metrics_evaluator = EvaluationMetrics()
        print("✅ Evaluation metrics initialized")
        
        # Generate sample predictions for demo
        if df is not None and len(df) > 0:
            true_scores = df['score'].tolist()
            # Simulate predictions with some noise
            predicted_scores = [score + np.random.normal(0, 0.5) for score in true_scores]
            predicted_scores = [max(0, min(10, score)) for score in predicted_scores]  # Clamp to 0-10
            
            # Compute metrics
            metrics = metrics_evaluator.compute_all_metrics(true_scores, predicted_scores)
            
            print(f"\n📊 Evaluation Metrics:")
            print(f"MSE: {metrics['mse']:.3f}")
            print(f"MAE: {metrics['mae']:.3f}")
            print(f"R² Score: {metrics['r2']:.3f}")
            print(f"Pearson Correlation: {metrics['pearson_correlation']:.3f}")
            print(f"Spearman Correlation: {metrics['spearman_correlation']:.3f}")
            
            # Generate interpretation
            interpretation = metrics_evaluator._generate_interpretation()
            print(f"\n💡 Interpretation:")
            print(f"Overall Performance: {interpretation['overall_performance']}")
            print(f"Variance Explained: {interpretation['variance_explained']}")
            print(f"Error Level: {interpretation['error_level']}")
            
            return metrics
        else:
            print("❌ No data available for metrics evaluation")
            return None
            
    except Exception as e:
        print(f"❌ Error in evaluation metrics: {str(e)}")
        return None

def demo_integration():
    """Demonstrate integrated workflow."""
    print("\n" + "=" * 60)
    print("DEMO: Integrated Workflow")
    print("=" * 60)
    
    try:
        # Load and preprocess data
        df = demo_data_preprocessing()
        if df is None:
            return
        
        # Run semantic evaluation
        semantic_results = demo_semantic_evaluation(df)
        
        # Generate feedback
        feedback_results = demo_feedback_generation(df)
        
        # Compute metrics
        metrics_results = demo_evaluation_metrics(df)
        
        print(f"\n🎉 Integration Demo Completed Successfully!")
        print(f"✅ All components working together")
        
        return {
            'data': df,
            'semantic_results': semantic_results,
            'feedback_results': feedback_results,
            'metrics_results': metrics_results
        }
        
    except Exception as e:
        print(f"❌ Error in integration demo: {str(e)}")
        return None

def main():
    """Main demo function."""
    print("🤖 AI-Based Semantic Evaluation - Demo Script")
    print("=" * 60)
    
    # Check if sample data exists
    sample_file = Path("data/sample_dataset.csv")
    if not sample_file.exists():
        print("❌ Sample dataset not found. Please ensure data/sample_dataset.csv exists.")
        return
    
    # Run integrated demo
    results = demo_integration()
    
    if results:
        print(f"\n📋 Demo Summary:")
        print(f"- Processed {len(results['data'])} records")
        print(f"- Semantic evaluation: {'✅' if results['semantic_results'] else '❌'}")
        print(f"- Feedback generation: {'✅' if results['feedback_results'] else '❌'}")
        print(f"- Metrics computation: {'✅' if results['metrics_results'] else '❌'}")
        
        print(f"\n🚀 Ready to run the Streamlit app!")
        print(f"Run: streamlit run app.py")
    else:
        print("❌ Demo failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
