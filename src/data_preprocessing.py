"""
Data Preprocessing Module for AI-Based Semantic Evaluation

This module handles loading, cleaning, and preprocessing of question-answer datasets
for semantic evaluation tasks.
"""

import pandas as pd
import numpy as np
import json
import re
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles data preprocessing for semantic evaluation tasks."""
    
    def __init__(self):
        self.data = None
        self.processed_data = None
        
    def load_data(self, file_path: str, file_type: str = 'csv') -> pd.DataFrame:
        """
        Load data from CSV or JSON file.
        
        Args:
            file_path: Path to the data file
            file_type: Type of file ('csv' or 'json')
            
        Returns:
            pandas DataFrame containing the loaded data
        """
        try:
            if file_type.lower() == 'csv':
                self.data = pd.read_csv(file_path)
            elif file_type.lower() == 'json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                self.data = pd.DataFrame(json_data)
            else:
                raise ValueError("File type must be 'csv' or 'json'")
                
            logger.info(f"Successfully loaded {len(self.data)} records from {file_path}")
            return self.data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def validate_data(self, required_columns: List[str] = None) -> bool:
        """
        Validate that the loaded data has required columns.
        
        Args:
            required_columns: List of required column names
            
        Returns:
            True if validation passes, False otherwise
        """
        if self.data is None:
            logger.error("No data loaded. Please load data first.")
            return False
            
        if required_columns is None:
            required_columns = ['question', 'model_answer', 'student_answer', 'score']
            
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
            
        logger.info("Data validation passed")
        return True
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text data.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text
        """
        if pd.isna(text) or text is None:
            return ""
            
        # Convert to string
        text = str(text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:()]', '', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def preprocess_data(self, 
                       text_columns: List[str] = None,
                       score_range: Tuple[float, float] = (0, 10)) -> pd.DataFrame:
        """
        Preprocess the loaded data.
        
        Args:
            text_columns: List of column names containing text to clean
            score_range: Tuple of (min_score, max_score) for validation
            
        Returns:
            Preprocessed DataFrame
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
            
        if text_columns is None:
            text_columns = ['question', 'model_answer', 'student_answer']
            
        # Create a copy for preprocessing
        self.processed_data = self.data.copy()
        
        # Clean text columns
        for col in text_columns:
            if col in self.processed_data.columns:
                self.processed_data[col] = self.processed_data[col].apply(self.clean_text)
                logger.info(f"Cleaned text column: {col}")
        
        # Validate and clean scores
        if 'score' in self.processed_data.columns:
            # Convert scores to numeric
            self.processed_data['score'] = pd.to_numeric(
                self.processed_data['score'], errors='coerce'
            )
            
            # Remove rows with invalid scores
            initial_count = len(self.processed_data)
            self.processed_data = self.processed_data.dropna(subset=['score'])
            
            # Validate score range
            invalid_scores = (
                (self.processed_data['score'] < score_range[0]) | 
                (self.processed_data['score'] > score_range[1])
            )
            
            if invalid_scores.any():
                logger.warning(f"Found {invalid_scores.sum()} scores outside range {score_range}")
                self.processed_data = self.processed_data[~invalid_scores]
            
            logger.info(f"Removed {initial_count - len(self.processed_data)} rows with invalid scores")
        
        # Remove rows with empty text
        for col in text_columns:
            if col in self.processed_data.columns:
                empty_mask = self.processed_data[col].str.len() == 0
                if empty_mask.any():
                    logger.warning(f"Found {empty_mask.sum()} empty entries in column {col}")
                    self.processed_data = self.processed_data[~empty_mask]
        
        logger.info(f"Preprocessing complete. Final dataset size: {len(self.processed_data)}")
        return self.processed_data
    
    def get_data_statistics(self) -> Dict:
        """
        Get statistics about the processed data.
        
        Returns:
            Dictionary containing data statistics
        """
        if self.processed_data is None:
            logger.error("No processed data available")
            return {}
            
        stats = {
            'total_records': len(self.processed_data),
            'columns': list(self.processed_data.columns),
            'text_columns': ['question', 'model_answer', 'student_answer'],
            'score_stats': {}
        }
        
        # Score statistics
        if 'score' in self.processed_data.columns:
            stats['score_stats'] = {
                'mean': float(self.processed_data['score'].mean()),
                'std': float(self.processed_data['score'].std()),
                'min': float(self.processed_data['score'].min()),
                'max': float(self.processed_data['score'].max()),
                'median': float(self.processed_data['score'].median())
            }
        
        # Text length statistics
        for col in stats['text_columns']:
            if col in self.processed_data.columns:
                lengths = self.processed_data[col].str.len()
                stats[f'{col}_length_stats'] = {
                    'mean': float(lengths.mean()),
                    'std': float(lengths.std()),
                    'min': int(lengths.min()),
                    'max': int(lengths.max()),
                    'median': float(lengths.median())
                }
        
        return stats
    
    def save_processed_data(self, output_path: str, file_type: str = 'csv') -> None:
        """
        Save processed data to file.
        
        Args:
            output_path: Path to save the processed data
            file_type: Type of file ('csv' or 'json')
        """
        if self.processed_data is None:
            logger.error("No processed data to save")
            return
            
        try:
            if file_type.lower() == 'csv':
                self.processed_data.to_csv(output_path, index=False)
            elif file_type.lower() == 'json':
                self.processed_data.to_json(output_path, orient='records', indent=2)
            else:
                raise ValueError("File type must be 'csv' or 'json'")
                
            logger.info(f"Processed data saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving processed data: {str(e)}")
            raise
    
    def create_sample_dataset(self, output_path: str = 'data/sample_dataset.csv') -> None:
        """
        Create a sample dataset for demonstration purposes.
        
        Args:
            output_path: Path to save the sample dataset
        """
        sample_data = {
            'question': [
                "Explain the concept of machine learning and its applications.",
                "What are the advantages and disadvantages of renewable energy?",
                "Describe the process of photosynthesis in plants.",
                "How does climate change affect biodiversity?",
                "Explain the principles of democracy and its importance."
            ],
            'model_answer': [
                "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed. It involves algorithms that can identify patterns in data and make predictions or classifications. Applications include image recognition, natural language processing, recommendation systems, autonomous vehicles, and medical diagnosis. The field continues to evolve with deep learning and neural networks enabling more complex tasks.",
                "Renewable energy sources like solar, wind, and hydroelectric power offer significant advantages including environmental sustainability, reduced greenhouse gas emissions, and decreased dependence on fossil fuels. They provide long-term energy security and can create jobs in emerging industries. However, disadvantages include high initial costs, intermittent availability requiring energy storage solutions, geographic limitations, and potential environmental impacts during construction. The transition requires substantial infrastructure investment and policy support.",
                "Photosynthesis is the process by which plants convert light energy into chemical energy. It occurs primarily in chloroplasts, specifically in the thylakoid membranes and stroma. The process involves two main stages: light-dependent reactions where chlorophyll absorbs light energy to split water molecules, producing oxygen and ATP, and light-independent reactions (Calvin cycle) where carbon dioxide is fixed and converted into glucose using ATP and NADPH. This process is essential for life on Earth as it produces oxygen and forms the base of most food chains.",
                "Climate change significantly impacts biodiversity through various mechanisms. Rising temperatures alter habitats and migration patterns, forcing species to adapt or relocate. Ocean acidification affects marine ecosystems, particularly coral reefs and shellfish. Changes in precipitation patterns disrupt ecosystems and food chains. Extreme weather events cause habitat destruction and species displacement. These factors contribute to species extinction, reduced genetic diversity, and ecosystem instability, ultimately threatening global biodiversity and ecosystem services.",
                "Democracy is a system of government where power is vested in the people, who exercise it directly or through elected representatives. Key principles include political equality, majority rule with minority rights, free and fair elections, rule of law, and protection of civil liberties. Democracy ensures accountability, promotes political participation, protects individual freedoms, and provides peaceful conflict resolution. It's important because it empowers citizens, prevents authoritarianism, encourages innovation and progress, and creates stable societies that respect human dignity and rights."
            ],
            'student_answer': [
                "Machine learning is when computers learn from data. It's used in many apps and websites to recommend things. It helps with recognizing images and understanding language. Companies use it to make better products and services.",
                "Renewable energy is good for the environment because it doesn't pollute. Solar panels and wind turbines are examples. But they cost a lot of money to install and sometimes don't work when there's no sun or wind. We need better batteries to store the energy.",
                "Plants make food using sunlight in a process called photosynthesis. They take in carbon dioxide and water, and with help from sunlight, they make glucose and release oxygen. This happens in the leaves where there are special cells called chloroplasts.",
                "Climate change makes the Earth warmer and affects animals and plants. Some species might go extinct because their homes are changing. The ocean is getting more acidic which hurts fish and coral. We need to protect nature better.",
                "Democracy means people choose their leaders by voting. Everyone should have equal rights and the government should follow laws. It's important because it gives people freedom and prevents dictators from taking over. It helps countries stay peaceful and prosperous."
            ],
            'score': [7.5, 6.0, 8.0, 6.5, 7.0]
        }
        
        # Create directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save sample data
        df = pd.DataFrame(sample_data)
        df.to_csv(output_path, index=False)
        logger.info(f"Sample dataset created at {output_path}")
        
        # Load and validate the sample data
        self.load_data(output_path)
        self.validate_data()
        self.preprocess_data()


def main():
    """Example usage of the DataPreprocessor class."""
    preprocessor = DataPreprocessor()
    
    # Create sample dataset
    preprocessor.create_sample_dataset()
    
    # Get statistics
    stats = preprocessor.get_data_statistics()
    print("Data Statistics:")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
