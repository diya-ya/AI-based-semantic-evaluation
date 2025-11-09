"""
Evaluation Metrics Module

This module implements comprehensive evaluation metrics including Pearson correlation,
MSE (Mean Squared Error), and other statistical measures for assessing model performance
in semantic answer evaluation.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score
)
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """Comprehensive evaluation metrics for semantic answer evaluation."""
    
    def __init__(self):
        self.results = {}
        self.plots_dir = Path("plots")
        self.plots_dir.mkdir(exist_ok=True)
    
    def compute_all_metrics(self, 
                           true_scores: List[float], 
                           predicted_scores: List[float],
                           similarities: Optional[List[float]] = None) -> Dict:
        """
        Compute all evaluation metrics.
        
        Args:
            true_scores: Ground truth scores
            predicted_scores: Model predicted scores
            similarities: Optional semantic similarity scores
            
        Returns:
            Dictionary containing all computed metrics
        """
        try:
            # Convert to numpy arrays
            true_scores = np.array(true_scores)
            predicted_scores = np.array(predicted_scores)
            
            if similarities is not None:
                similarities = np.array(similarities)
            
            # Basic regression metrics
            metrics = {
                'mse': mean_squared_error(true_scores, predicted_scores),
                'rmse': np.sqrt(mean_squared_error(true_scores, predicted_scores)),
                'mae': mean_absolute_error(true_scores, predicted_scores),
                'r2': r2_score(true_scores, predicted_scores),
                'mape': self._compute_mape(true_scores, predicted_scores)
            }
            
            # Correlation metrics
            pearson_corr, pearson_p = pearsonr(true_scores, predicted_scores)
            spearman_corr, spearman_p = spearmanr(true_scores, predicted_scores)
            
            metrics.update({
                'pearson_correlation': pearson_corr,
                'pearson_p_value': pearson_p,
                'spearman_correlation': spearman_corr,
                'spearman_p_value': spearman_p
            })
            
            # Score distribution analysis
            metrics.update(self._analyze_score_distributions(true_scores, predicted_scores))
            
            # Error analysis
            metrics.update(self._analyze_errors(true_scores, predicted_scores))
            
            # Semantic similarity analysis (if provided)
            if similarities is not None:
                metrics.update(self._analyze_similarity_correlation(true_scores, predicted_scores, similarities))
            
            # Performance by score range
            metrics.update(self._analyze_performance_by_range(true_scores, predicted_scores))
            
            self.results = metrics
            logger.info("All evaluation metrics computed successfully")
            return metrics
            
        except Exception as e:
            logger.error(f"Error computing metrics: {str(e)}")
            raise
    
    def _compute_mape(self, true_scores: np.ndarray, predicted_scores: np.ndarray) -> float:
        """Compute Mean Absolute Percentage Error."""
        # Avoid division by zero
        mask = true_scores != 0
        if not np.any(mask):
            return float('inf')
        
        return np.mean(np.abs((true_scores[mask] - predicted_scores[mask]) / true_scores[mask])) * 100
    
    def _analyze_score_distributions(self, true_scores: np.ndarray, predicted_scores: np.ndarray) -> Dict:
        """Analyze score distributions."""
        return {
            'true_scores_stats': {
                'mean': float(np.mean(true_scores)),
                'std': float(np.std(true_scores)),
                'min': float(np.min(true_scores)),
                'max': float(np.max(true_scores)),
                'median': float(np.median(true_scores)),
                'q25': float(np.percentile(true_scores, 25)),
                'q75': float(np.percentile(true_scores, 75))
            },
            'predicted_scores_stats': {
                'mean': float(np.mean(predicted_scores)),
                'std': float(np.std(predicted_scores)),
                'min': float(np.min(predicted_scores)),
                'max': float(np.max(predicted_scores)),
                'median': float(np.median(predicted_scores)),
                'q25': float(np.percentile(predicted_scores, 25)),
                'q75': float(np.percentile(predicted_scores, 75))
            }
        }
    
    def _analyze_errors(self, true_scores: np.ndarray, predicted_scores: np.ndarray) -> Dict:
        """Analyze prediction errors."""
        errors = predicted_scores - true_scores
        
        return {
            'error_stats': {
                'mean_error': float(np.mean(errors)),
                'std_error': float(np.std(errors)),
                'min_error': float(np.min(errors)),
                'max_error': float(np.max(errors)),
                'median_error': float(np.median(errors))
            },
            'error_distribution': {
                'overestimated': int(np.sum(errors > 0)),
                'underestimated': int(np.sum(errors < 0)),
                'exact_matches': int(np.sum(errors == 0))
            }
        }
    
    def _analyze_similarity_correlation(self, true_scores: np.ndarray, 
                                       predicted_scores: np.ndarray, 
                                       similarities: np.ndarray) -> Dict:
        """Analyze correlation with semantic similarities."""
        similarity_true_corr, similarity_true_p = pearsonr(similarities, true_scores)
        similarity_pred_corr, similarity_pred_p = pearsonr(similarities, predicted_scores)
        
        return {
            'similarity_correlations': {
                'similarity_vs_true_scores': {
                    'correlation': similarity_true_corr,
                    'p_value': similarity_true_p
                },
                'similarity_vs_predicted_scores': {
                    'correlation': similarity_pred_corr,
                    'p_value': similarity_pred_p
                }
            }
        }
    
    def _analyze_performance_by_range(self, true_scores: np.ndarray, predicted_scores: np.ndarray) -> Dict:
        """Analyze performance across different score ranges."""
        ranges = [(0, 3), (3, 6), (6, 8), (8, 10)]
        performance_by_range = {}
        
        for low, high in ranges:
            mask = (true_scores >= low) & (true_scores < high)
            if np.any(mask):
                range_true = true_scores[mask]
                range_pred = predicted_scores[mask]
                
                performance_by_range[f'score_range_{low}_{high}'] = {
                    'count': int(np.sum(mask)),
                    'mse': float(mean_squared_error(range_true, range_pred)),
                    'mae': float(mean_absolute_error(range_true, range_pred)),
                    'r2': float(r2_score(range_true, range_pred)),
                    'pearson_corr': float(pearsonr(range_true, range_pred)[0])
                }
        
        return {'performance_by_range': performance_by_range}
    
    def create_visualizations(self, true_scores: List[float], predicted_scores: List[float],
                             similarities: Optional[List[float]] = None) -> Dict[str, str]:
        """
        Create visualization plots for evaluation results.
        
        Args:
            true_scores: Ground truth scores
            predicted_scores: Model predicted scores
            similarities: Optional semantic similarity scores
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        try:
            plot_paths = {}
            
            # Set style
            plt.style.use('seaborn-v0_8')
            
            # 1. Scatter plot: True vs Predicted
            plt.figure(figsize=(10, 8))
            plt.scatter(true_scores, predicted_scores, alpha=0.6, s=50)
            plt.plot([0, 10], [0, 10], 'r--', label='Perfect Prediction')
            plt.xlabel('True Scores')
            plt.ylabel('Predicted Scores')
            plt.title('True vs Predicted Scores')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Add correlation text
            corr = pearsonr(true_scores, predicted_scores)[0]
            plt.text(0.05, 0.95, f'Pearson r = {corr:.3f}', 
                    transform=plt.gca().transAxes, fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            scatter_path = self.plots_dir / 'true_vs_predicted.png'
            plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths['scatter_plot'] = str(scatter_path)
            
            # 2. Residual plot
            plt.figure(figsize=(10, 6))
            residuals = np.array(predicted_scores) - np.array(true_scores)
            plt.scatter(true_scores, residuals, alpha=0.6)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('True Scores')
            plt.ylabel('Residuals (Predicted - True)')
            plt.title('Residual Plot')
            plt.grid(True, alpha=0.3)
            
            residual_path = self.plots_dir / 'residual_plot.png'
            plt.savefig(residual_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths['residual_plot'] = str(residual_path)
            
            # 3. Score distribution comparison
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            plt.hist(true_scores, bins=20, alpha=0.7, label='True Scores', color='blue')
            plt.xlabel('Score')
            plt.ylabel('Frequency')
            plt.title('True Scores Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            plt.hist(predicted_scores, bins=20, alpha=0.7, label='Predicted Scores', color='orange')
            plt.xlabel('Score')
            plt.ylabel('Frequency')
            plt.title('Predicted Scores Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            distribution_path = self.plots_dir / 'score_distributions.png'
            plt.savefig(distribution_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths['distribution_plot'] = str(distribution_path)
            
            # 4. Error distribution
            plt.figure(figsize=(10, 6))
            plt.hist(residuals, bins=20, alpha=0.7, color='green')
            plt.xlabel('Prediction Error')
            plt.ylabel('Frequency')
            plt.title('Error Distribution')
            plt.axvline(x=0, color='r', linestyle='--', label='Perfect Prediction')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            error_path = self.plots_dir / 'error_distribution.png'
            plt.savefig(error_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths['error_plot'] = str(error_path)
            
            # 5. Similarity correlation plot (if similarities provided)
            if similarities is not None:
                plt.figure(figsize=(12, 5))
                
                plt.subplot(1, 2, 1)
                plt.scatter(similarities, true_scores, alpha=0.6, color='blue')
                plt.xlabel('Semantic Similarity')
                plt.ylabel('True Scores')
                plt.title('Similarity vs True Scores')
                plt.grid(True, alpha=0.3)
                
                plt.subplot(1, 2, 2)
                plt.scatter(similarities, predicted_scores, alpha=0.6, color='orange')
                plt.xlabel('Semantic Similarity')
                plt.ylabel('Predicted Scores')
                plt.title('Similarity vs Predicted Scores')
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                similarity_path = self.plots_dir / 'similarity_correlations.png'
                plt.savefig(similarity_path, dpi=300, bbox_inches='tight')
                plt.close()
                plot_paths['similarity_plot'] = str(similarity_path)
            
            logger.info(f"Created {len(plot_paths)} visualization plots")
            return plot_paths
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
            raise
    
    def generate_report(self, output_path: str = 'evaluation_report.json') -> None:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            output_path: Path to save the report
        """
        if not self.results:
            logger.error("No evaluation results available. Run compute_all_metrics first.")
            return
        
        try:
            # Create comprehensive report
            report = {
                'evaluation_summary': {
                    'total_samples': len(self.results.get('true_scores_stats', {}).get('mean', 0)),
                    'evaluation_date': pd.Timestamp.now().isoformat(),
                    'model_performance': {
                        'mse': self.results.get('mse', 0),
                        'rmse': self.results.get('rmse', 0),
                        'mae': self.results.get('mae', 0),
                        'r2': self.results.get('r2', 0),
                        'pearson_correlation': self.results.get('pearson_correlation', 0),
                        'spearman_correlation': self.results.get('spearman_correlation', 0)
                    }
                },
                'detailed_metrics': self.results,
                'interpretation': self._generate_interpretation()
            }
            
            # Save report
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Evaluation report saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            raise
    
    def _generate_interpretation(self) -> Dict:
        """Generate interpretation of the evaluation results."""
        if not self.results:
            return {}
        
        interpretation = {}
        
        # Overall performance interpretation
        pearson_corr = self.results.get('pearson_correlation', 0)
        r2 = self.results.get('r2', 0)
        mae = self.results.get('mae', 0)
        
        if pearson_corr >= 0.8:
            interpretation['overall_performance'] = "Excellent model performance with strong correlation to human scores."
        elif pearson_corr >= 0.6:
            interpretation['overall_performance'] = "Good model performance with moderate correlation to human scores."
        elif pearson_corr >= 0.4:
            interpretation['overall_performance'] = "Fair model performance with weak correlation to human scores."
        else:
            interpretation['overall_performance'] = "Poor model performance with very weak correlation to human scores."
        
        # R² interpretation
        if r2 >= 0.7:
            interpretation['variance_explained'] = f"Model explains {r2:.1%} of variance in scores - excellent fit."
        elif r2 >= 0.5:
            interpretation['variance_explained'] = f"Model explains {r2:.1%} of variance in scores - good fit."
        elif r2 >= 0.3:
            interpretation['variance_explained'] = f"Model explains {r2:.1%} of variance in scores - moderate fit."
        else:
            interpretation['variance_explained'] = f"Model explains {r2:.1%} of variance in scores - poor fit."
        
        # Error interpretation
        if mae <= 1.0:
            interpretation['error_level'] = f"Mean absolute error of {mae:.2f} indicates high accuracy."
        elif mae <= 2.0:
            interpretation['error_level'] = f"Mean absolute error of {mae:.2f} indicates moderate accuracy."
        else:
            interpretation['error_level'] = f"Mean absolute error of {mae:.2f} indicates low accuracy."
        
        return interpretation
    
    def compare_models(self, model_results: Dict[str, Dict]) -> Dict:
        """
        Compare performance of multiple models.
        
        Args:
            model_results: Dictionary mapping model names to their evaluation results
            
        Returns:
            Comparison results
        """
        try:
            comparison = {}
            
            for model_name, results in model_results.items():
                comparison[model_name] = {
                    'mse': results.get('mse', 0),
                    'mae': results.get('mae', 0),
                    'r2': results.get('r2', 0),
                    'pearson_correlation': results.get('pearson_correlation', 0)
                }
            
            # Find best model for each metric
            best_models = {}
            for metric in ['mse', 'mae', 'r2', 'pearson_correlation']:
                if metric in ['mse', 'mae']:
                    # Lower is better
                    best_model = min(model_results.keys(), 
                                   key=lambda x: model_results[x].get(metric, float('inf')))
                else:
                    # Higher is better
                    best_model = max(model_results.keys(), 
                                   key=lambda x: model_results[x].get(metric, 0))
                
                best_models[metric] = best_model
            
            comparison['best_models'] = best_models
            
            logger.info("Model comparison completed")
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing models: {str(e)}")
            raise


def main():
    """Example usage of the EvaluationMetrics class."""
    # Generate sample data
    np.random.seed(42)
    true_scores = np.random.uniform(0, 10, 100)
    predicted_scores = true_scores + np.random.normal(0, 1, 100)
    similarities = np.random.uniform(0, 1, 100)
    
    # Initialize evaluator
    evaluator = EvaluationMetrics()
    
    # Compute metrics
    metrics = evaluator.compute_all_metrics(true_scores, predicted_scores, similarities)
    
    # Create visualizations
    plot_paths = evaluator.create_visualizations(true_scores, predicted_scores, similarities)
    
    # Generate report
    evaluator.generate_report()
    
    print("Evaluation completed successfully!")
    print(f"Pearson Correlation: {metrics['pearson_correlation']:.3f}")
    print(f"R² Score: {metrics['r2']:.3f}")
    print(f"MSE: {metrics['mse']:.3f}")


if __name__ == "__main__":
    main()
