"""
BERT-based Scoring Model for Answer Evaluation

This module implements a fine-tuned BERT/RoBERTa model for predicting numeric scores
between 0-10 for student answers based on semantic similarity and quality assessment.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoConfig,
    get_linear_schedule_with_warmup
)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import json
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnswerDataset(Dataset):
    """Custom dataset for answer evaluation."""
    
    def __init__(self, questions: List[str], model_answers: List[str], 
                 student_answers: List[str], scores: List[float], 
                 tokenizer, max_length: int = 512):
        """
        Initialize the dataset.
        
        Args:
            questions: List of questions
            model_answers: List of model answers
            student_answers: List of student answers
            scores: List of scores
            tokenizer: Tokenizer for the model
            max_length: Maximum sequence length
        """
        self.questions = questions
        self.model_answers = model_answers
        self.student_answers = student_answers
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question = str(self.questions[idx])
        model_answer = str(self.model_answers[idx])
        student_answer = str(self.student_answers[idx])
        score = float(self.scores[idx])
        
        # Create input text combining question, model answer, and student answer
        input_text = f"[QUESTION] {question} [MODEL] {model_answer} [STUDENT] {student_answer}"
        
        # Tokenize
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(score, dtype=torch.float)
        }


class BERTScoringModel(nn.Module):
    """BERT-based model for answer scoring."""
    
    def __init__(self, model_name: str = 'bert-base-uncased', num_classes: int = 1, 
                 dropout_rate: float = 0.3):
        """
        Initialize the BERT scoring model.
        
        Args:
            model_name: Name of the pre-trained BERT model
            num_classes: Number of output classes (1 for regression)
            dropout_rate: Dropout rate for regularization
        """
        super(BERTScoringModel, self).__init__()
        
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        """Forward pass of the model."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.classifier(output)


class BERTScorer:
    """Main class for BERT-based answer scoring."""
    
    def __init__(self, model_name: str = 'bert-base-uncased', device: str = 'auto'):
        """
        Initialize the BERT scorer.
        
        Args:
            model_name: Name of the pre-trained model
            device: Device to run the model on
        """
        self.model_name = model_name
        self.device = self._get_device(device)
        self.tokenizer = None
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        self._load_tokenizer()
        self._initialize_model()
    
    def _get_device(self, device: str) -> str:
        """Determine the appropriate device."""
        if device == 'auto':
            if torch.cuda.is_available():
                return 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'
            else:
                return 'cpu'
        return device
    
    def _load_tokenizer(self):
        """Load the tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            logger.info(f"Tokenizer loaded for {self.model_name}")
        except Exception as e:
            logger.error(f"Error loading tokenizer: {str(e)}")
            raise
    
    def _initialize_model(self):
        """Initialize the BERT model."""
        try:
            self.model = BERTScoringModel(self.model_name).to(self.device)
            logger.info(f"BERT model initialized on {self.device}")
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Prepare data for training, validation, and testing.
        
        Args:
            df: DataFrame with columns ['question', 'model_answer', 'student_answer', 'score']
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Split data
        train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
        
        logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        # Create datasets
        train_dataset = AnswerDataset(
            train_df['question'].tolist(),
            train_df['model_answer'].tolist(),
            train_df['student_answer'].tolist(),
            train_df['score'].tolist(),
            self.tokenizer
        )
        
        val_dataset = AnswerDataset(
            val_df['question'].tolist(),
            val_df['model_answer'].tolist(),
            val_df['student_answer'].tolist(),
            val_df['score'].tolist(),
            self.tokenizer
        )
        
        test_dataset = AnswerDataset(
            test_df['question'].tolist(),
            test_df['model_answer'].tolist(),
            test_df['student_answer'].tolist(),
            test_df['score'].tolist(),
            self.tokenizer
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int = 5, learning_rate: float = 2e-5) -> Dict:
        """
        Train the BERT scoring model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            learning_rate: Learning rate for optimization
            
        Returns:
            Dictionary containing training history
        """
        try:
            # Set up training
            criterion = nn.MSELoss()
            optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
            
            # Learning rate scheduler
            total_steps = len(train_loader) * epochs
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=0,
                num_training_steps=total_steps
            )
            
            # Training history
            history = {
                'train_loss': [],
                'val_loss': [],
                'val_mse': [],
                'val_mae': [],
                'val_r2': []
            }
            
            best_val_loss = float('inf')
            
            for epoch in range(epochs):
                # Training phase
                self.model.train()
                train_loss = 0
                
                for batch in train_loader:
                    optimizer.zero_grad()
                    
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = self.model(input_ids, attention_mask)
                    loss = criterion(outputs.squeeze(), labels)
                    
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    
                    train_loss += loss.item()
                
                # Validation phase
                val_loss, val_metrics = self._evaluate(val_loader, criterion)
                
                # Update history
                history['train_loss'].append(train_loss / len(train_loader))
                history['val_loss'].append(val_loss)
                history['val_mse'].append(val_metrics['mse'])
                history['val_mae'].append(val_metrics['mae'])
                history['val_r2'].append(val_metrics['r2'])
                
                logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss/len(train_loader):.4f}, "
                           f"Val Loss: {val_loss:.4f}, Val R²: {val_metrics['r2']:.4f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save_model('models/best_bert_scorer.pth')
            
            self.is_trained = True
            logger.info("Training completed successfully")
            return history
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
    
    def _evaluate(self, data_loader: DataLoader, criterion) -> Tuple[float, Dict]:
        """Evaluate the model on a dataset."""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs.squeeze(), labels)
                
                total_loss += loss.item()
                all_predictions.extend(outputs.squeeze().cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        predictions = np.array(all_predictions)
        labels = np.array(all_labels)
        
        metrics = {
            'mse': mean_squared_error(labels, predictions),
            'mae': mean_absolute_error(labels, predictions),
            'r2': r2_score(labels, predictions)
        }
        
        return total_loss / len(data_loader), metrics
    
    def predict(self, question: str, model_answer: str, student_answer: str) -> float:
        """
        Predict score for a single answer pair.
        
        Args:
            question: The question
            model_answer: The model answer
            student_answer: The student answer
            
        Returns:
            Predicted score
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            self.model.eval()
            
            # Prepare input
            input_text = f"[QUESTION] {question} [MODEL] {model_answer} [STUDENT] {student_answer}"
            
            encoding = self.tokenizer(
                input_text,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            with torch.no_grad():
                output = self.model(input_ids, attention_mask)
                score = output.squeeze().item()
            
            # Ensure score is within valid range
            score = max(0, min(10, score))
            
            return score
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise
    
    def batch_predict(self, questions: List[str], model_answers: List[str], 
                     student_answers: List[str]) -> List[float]:
        """
        Predict scores for multiple answer pairs.
        
        Args:
            questions: List of questions
            model_answers: List of model answers
            student_answers: List of student answers
            
        Returns:
            List of predicted scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            self.model.eval()
            predictions = []
            
            for q, ma, sa in zip(questions, model_answers, student_answers):
                score = self.predict(q, ma, sa)
                predictions.append(score)
            
            logger.info(f"Made predictions for {len(predictions)} answer pairs")
            return predictions
            
        except Exception as e:
            logger.error(f"Error in batch prediction: {str(e)}")
            raise
    
    def _save_model(self, path: str):
        """Save the trained model."""
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'model_name': self.model_name,
                'is_trained': self.is_trained
            }, path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, path: str):
        """Load a trained model."""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.is_trained = checkpoint['is_trained']
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict:
        """Get information about the current model."""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'is_trained': self.is_trained,
            'num_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }


def main():
    """Example usage of the BERTScorer class."""
    # This would typically be used with actual training data
    scorer = BERTScorer()
    
    # Example prediction (would need trained model)
    question = "Explain the concept of machine learning."
    model_answer = "Machine learning is a subset of AI that enables computers to learn from data."
    student_answer = "Machine learning is when computers learn from data automatically."
    
    print("BERT Scorer initialized successfully")
    print(f"Model info: {scorer.get_model_info()}")


if __name__ == "__main__":
    main()
