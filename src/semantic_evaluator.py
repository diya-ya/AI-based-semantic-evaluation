"""
Semantic Similarity Evaluation Module

This module implements semantic similarity evaluation using Sentence-BERT embeddings
and cosine similarity to assess the semantic closeness between model and student answers.
"""

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemanticEvaluator:
    """Handles semantic similarity evaluation using Sentence-BERT embeddings."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: str = 'auto'):
        """
        Initialize the semantic evaluator.
        
        Args:
            model_name: Name of the Sentence-BERT model to use
            device: Device to run the model on ('cpu', 'cuda', or 'auto')
        """
        self.model_name = model_name
        self.device = self._get_device(device)
        self.model = None
        self._load_model()
        
    def _get_device(self, device: str) -> str:
        """Determine the appropriate device for model execution."""
        if device == 'auto':
            if torch.cuda.is_available():
                return 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'
            else:
                return 'cpu'
        return device
    
    def _load_model(self):
        """Load the Sentence-BERT model."""
        try:
            logger.info(f"Loading Sentence-BERT model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"Model loaded successfully on device: {self.device}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode texts into embeddings using Sentence-BERT.
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding
            
        Returns:
            numpy array of embeddings
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please initialize the evaluator.")
            
        try:
            embeddings = self.model.encode(
                texts, 
                batch_size=batch_size,
                show_progress_bar=len(texts) > 10,
                convert_to_numpy=True
            )
            logger.info(f"Encoded {len(texts)} texts into embeddings")
            return embeddings
        except Exception as e:
            logger.error(f"Error encoding texts: {str(e)}")
            raise
    
    def compute_similarity(self, 
                          model_answer: str, 
                          student_answer: str) -> float:
        """
        Compute cosine similarity between model and student answers.
        
        Args:
            model_answer: The model/teacher answer
            student_answer: The student answer
            
        Returns:
            Cosine similarity score between 0 and 1
        """
        try:
            # Encode both answers
            embeddings = self.encode_texts([model_answer, student_answer])
            
            # Compute cosine similarity
            similarity = cosine_similarity(
                embeddings[0].reshape(1, -1), 
                embeddings[1].reshape(1, -1)
            )[0][0]
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error computing similarity: {str(e)}")
            raise
    
    def batch_compute_similarity(self, 
                                model_answers: List[str], 
                                student_answers: List[str]) -> List[float]:
        """
        Compute cosine similarity for multiple answer pairs.
        
        Args:
            model_answers: List of model/teacher answers
            student_answers: List of student answers
            
        Returns:
            List of cosine similarity scores
        """
        if len(model_answers) != len(student_answers):
            raise ValueError("Number of model and student answers must be equal")
            
        try:
            # Combine all texts for batch encoding
            all_texts = model_answers + student_answers
            
            # Encode all texts
            embeddings = self.encode_texts(all_texts)
            
            # Split embeddings
            model_embeddings = embeddings[:len(model_answers)]
            student_embeddings = embeddings[len(model_answers):]
            
            # Compute similarities
            similarities = []
            for model_emb, student_emb in zip(model_embeddings, student_embeddings):
                similarity = cosine_similarity(
                    model_emb.reshape(1, -1), 
                    student_emb.reshape(1, -1)
                )[0][0]
                similarities.append(float(similarity))
            
            logger.info(f"Computed similarities for {len(model_answers)} answer pairs")
            return similarities
            
        except Exception as e:
            logger.error(f"Error in batch similarity computation: {str(e)}")
            raise
    
    def evaluate_dataset(self, 
                        questions: List[str],
                        model_answers: List[str], 
                        student_answers: List[str],
                        include_question_context: bool = True) -> Dict:
        """
        Evaluate semantic similarity for an entire dataset.
        
        Args:
            questions: List of questions
            model_answers: List of model answers
            student_answers: List of student answers
            include_question_context: Whether to include question in similarity computation
            
        Returns:
            Dictionary containing evaluation results
        """
        if not all(len(lst) == len(questions) for lst in [model_answers, student_answers]):
            raise ValueError("All input lists must have the same length")
            
        try:
            results = {
                'similarities': [],
                'statistics': {},
                'detailed_results': []
            }
            
            if include_question_context:
                # Include question context in similarity computation
                combined_model = [f"Q: {q}\nA: {a}" for q, a in zip(questions, model_answers)]
                combined_student = [f"Q: {q}\nA: {a}" for q, a in zip(questions, student_answers)]
                
                similarities = self.batch_compute_similarity(combined_model, combined_student)
            else:
                # Compute similarity only between answers
                similarities = self.batch_compute_similarity(model_answers, student_answers)
            
            results['similarities'] = similarities
            
            # Compute statistics
            similarities_array = np.array(similarities)
            results['statistics'] = {
                'mean': float(np.mean(similarities_array)),
                'std': float(np.std(similarities_array)),
                'min': float(np.min(similarities_array)),
                'max': float(np.max(similarities_array)),
                'median': float(np.median(similarities_array)),
                'q25': float(np.percentile(similarities_array, 25)),
                'q75': float(np.percentile(similarities_array, 75))
            }
            
            # Create detailed results
            for i, (q, ma, sa, sim) in enumerate(zip(questions, model_answers, student_answers, similarities)):
                results['detailed_results'].append({
                    'index': i,
                    'question': q,
                    'model_answer': ma,
                    'student_answer': sa,
                    'similarity': sim
                })
            
            logger.info(f"Evaluated {len(questions)} question-answer pairs")
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating dataset: {str(e)}")
            raise
    
    def get_similarity_interpretation(self, similarity_score: float) -> str:
        """
        Provide interpretation of similarity score.
        
        Args:
            similarity_score: Cosine similarity score between 0 and 1
            
        Returns:
            String interpretation of the score
        """
        if similarity_score >= 0.9:
            return "Very High Similarity - Excellent semantic match"
        elif similarity_score >= 0.8:
            return "High Similarity - Strong semantic alignment"
        elif similarity_score >= 0.7:
            return "Good Similarity - Substantial semantic overlap"
        elif similarity_score >= 0.6:
            return "Moderate Similarity - Some semantic alignment"
        elif similarity_score >= 0.5:
            return "Low Similarity - Limited semantic overlap"
        elif similarity_score >= 0.3:
            return "Very Low Similarity - Minimal semantic connection"
        else:
            return "Poor Similarity - Very little semantic alignment"
    
    def save_embeddings(self, texts: List[str], output_path: str) -> None:
        """
        Save embeddings to file for future use.
        
        Args:
            texts: List of texts to encode and save
            output_path: Path to save the embeddings
        """
        try:
            embeddings = self.encode_texts(texts)
            
            # Create directory if it doesn't exist
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save embeddings
            np.save(output_path, embeddings)
            logger.info(f"Embeddings saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving embeddings: {str(e)}")
            raise
    
    def load_embeddings(self, input_path: str) -> np.ndarray:
        """
        Load embeddings from file.
        
        Args:
            input_path: Path to the saved embeddings
            
        Returns:
            numpy array of embeddings
        """
        try:
            embeddings = np.load(input_path)
            logger.info(f"Embeddings loaded from {input_path}")
            return embeddings
        except Exception as e:
            logger.error(f"Error loading embeddings: {str(e)}")
            raise


class AdvancedSemanticEvaluator(SemanticEvaluator):
    """Extended semantic evaluator with additional features."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: str = 'auto'):
        super().__init__(model_name, device)
    
    def compute_multiple_similarities(self, 
                                    model_answer: str, 
                                    student_answer: str,
                                    include_sentence_level: bool = True) -> Dict:
        """
        Compute multiple types of semantic similarities.
        
        Args:
            model_answer: The model answer
            student_answer: The student answer
            include_sentence_level: Whether to compute sentence-level similarities
            
        Returns:
            Dictionary with different similarity metrics
        """
        try:
            results = {}
            
            # Overall similarity
            results['overall_similarity'] = self.compute_similarity(model_answer, student_answer)
            
            if include_sentence_level:
                # Sentence-level similarities
                model_sentences = model_answer.split('. ')
                student_sentences = student_answer.split('. ')
                
                if len(model_sentences) > 1 and len(student_sentences) > 1:
                    # Compute similarity for each sentence pair
                    sentence_similarities = []
                    for ms in model_sentences:
                        for ss in student_sentences:
                            if ms.strip() and ss.strip():
                                sim = self.compute_similarity(ms.strip(), ss.strip())
                                sentence_similarities.append(sim)
                    
                    if sentence_similarities:
                        results['sentence_similarities'] = {
                            'mean': float(np.mean(sentence_similarities)),
                            'max': float(np.max(sentence_similarities)),
                            'min': float(np.min(sentence_similarities)),
                            'all': sentence_similarities
                        }
            
            # Length-based analysis
            results['length_analysis'] = {
                'model_length': len(model_answer.split()),
                'student_length': len(student_answer.split()),
                'length_ratio': len(student_answer.split()) / max(len(model_answer.split()), 1)
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error computing multiple similarities: {str(e)}")
            raise


def main():
    """Example usage of the SemanticEvaluator class."""
    evaluator = SemanticEvaluator()
    
    # Example texts
    model_answer = "Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed."
    student_answer = "Machine learning is when computers learn from data automatically without human programming."
    
    # Compute similarity
    similarity = evaluator.compute_similarity(model_answer, student_answer)
    interpretation = evaluator.get_similarity_interpretation(similarity)
    
    print(f"Similarity Score: {similarity:.3f}")
    print(f"Interpretation: {interpretation}")


if __name__ == "__main__":
    main()
