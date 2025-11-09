"""
Feedback Generation Module

This module implements feedback generation using language models (GPT, Llama, or local models)
to provide detailed textual feedback on student answers covering coverage, relevance, grammar, and coherence.
"""

import openai
import requests
import json
import re
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeedbackGenerator:
    """Base class for feedback generation."""
    
    def __init__(self):
        self.is_configured = False
    
    def generate_feedback(self, question: str, model_answer: str, 
                         student_answer: str, score: float) -> Dict:
        """
        Generate comprehensive feedback for a student answer.
        
        Args:
            question: The question
            model_answer: The model/teacher answer
            student_answer: The student answer
            score: The predicted score (0-10)
            
        Returns:
            Dictionary containing feedback components
        """
        raise NotImplementedError("Subclasses must implement generate_feedback method")
    
    def _analyze_coverage(self, model_answer: str, student_answer: str) -> Tuple[str, float]:
        """Analyze topic coverage."""
        # Simple keyword-based coverage analysis
        model_words = set(model_answer.lower().split())
        student_words = set(student_answer.lower().split())
        
        # Remove common words
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        model_words = model_words - common_words
        student_words = student_words - common_words
        
        coverage_ratio = len(student_words.intersection(model_words)) / max(len(model_words), 1)
        
        if coverage_ratio >= 0.8:
            coverage_feedback = "Excellent topic coverage. The answer addresses most key points comprehensively."
        elif coverage_ratio >= 0.6:
            coverage_feedback = "Good topic coverage. The answer covers most important aspects."
        elif coverage_ratio >= 0.4:
            coverage_feedback = "Moderate topic coverage. Some key points are missing or underdeveloped."
        else:
            coverage_feedback = "Limited topic coverage. Many important aspects are not addressed."
        
        return coverage_feedback, coverage_ratio
    
    def _analyze_relevance(self, question: str, student_answer: str) -> Tuple[str, float]:
        """Analyze answer relevance to the question."""
        # Simple relevance analysis based on question keywords
        question_words = set(re.findall(r'\b\w+\b', question.lower()))
        answer_words = set(re.findall(r'\b\w+\b', student_answer.lower()))
        
        # Remove common words
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'what', 'how', 'why', 'when', 'where', 'who', 'which'}
        question_words = question_words - common_words
        answer_words = answer_words - common_words
        
        relevance_ratio = len(answer_words.intersection(question_words)) / max(len(question_words), 1)
        
        if relevance_ratio >= 0.7:
            relevance_feedback = "Highly relevant answer that directly addresses the question."
        elif relevance_ratio >= 0.5:
            relevance_feedback = "Generally relevant answer with good connection to the question."
        elif relevance_ratio >= 0.3:
            relevance_feedback = "Somewhat relevant but could be more focused on the specific question."
        else:
            relevance_feedback = "Answer lacks relevance to the specific question asked."
        
        return relevance_feedback, relevance_ratio
    
    def _analyze_grammar(self, student_answer: str) -> Tuple[str, float]:
        """Analyze grammar and language quality."""
        # Simple grammar analysis
        sentences = re.split(r'[.!?]+', student_answer)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return "No complete sentences found.", 0.0
        
        # Basic grammar checks
        grammar_score = 1.0
        
        # Check for capitalization
        proper_caps = sum(1 for s in sentences if s and s[0].isupper())
        grammar_score *= proper_caps / len(sentences)
        
        # Check for sentence length variety
        avg_length = sum(len(s.split()) for s in sentences) / len(sentences)
        if 5 <= avg_length <= 25:
            grammar_score *= 1.0
        elif 3 <= avg_length <= 30:
            grammar_score *= 0.8
        else:
            grammar_score *= 0.6
        
        # Check for basic punctuation
        has_punctuation = any(s.endswith(('.', '!', '?')) for s in sentences)
        if has_punctuation:
            grammar_score *= 1.0
        else:
            grammar_score *= 0.7
        
        if grammar_score >= 0.8:
            grammar_feedback = "Good grammar and sentence structure with proper punctuation."
        elif grammar_score >= 0.6:
            grammar_feedback = "Generally good grammar with minor issues in structure or punctuation."
        elif grammar_score >= 0.4:
            grammar_feedback = "Some grammar issues that affect readability and clarity."
        else:
            grammar_feedback = "Significant grammar and structure issues that hinder understanding."
        
        return grammar_feedback, grammar_score
    
    def _analyze_coherence(self, student_answer: str) -> Tuple[str, float]:
        """Analyze answer coherence and flow."""
        sentences = re.split(r'[.!?]+', student_answer)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return "Answer is too short to assess coherence.", 0.5
        
        # Simple coherence analysis based on transition words and sentence flow
        transition_words = ['however', 'therefore', 'moreover', 'furthermore', 'additionally', 'consequently', 'meanwhile', 'similarly', 'in contrast', 'on the other hand', 'first', 'second', 'third', 'finally', 'in conclusion', 'to summarize']
        
        coherence_score = 0.5  # Base score
        
        # Check for transition words
        answer_lower = student_answer.lower()
        transition_count = sum(1 for word in transition_words if word in answer_lower)
        coherence_score += min(transition_count * 0.1, 0.3)
        
        # Check for logical flow (simple heuristic)
        if len(sentences) >= 3:
            coherence_score += 0.2
        
        if coherence_score >= 0.8:
            coherence_feedback = "Excellent coherence with clear logical flow and good transitions."
        elif coherence_score >= 0.6:
            coherence_feedback = "Good coherence with generally clear organization."
        elif coherence_score >= 0.4:
            coherence_feedback = "Moderate coherence but could benefit from better organization."
        else:
            coherence_feedback = "Poor coherence with unclear organization and flow."
        
        return coherence_feedback, coherence_score


class OpenAIFeedbackGenerator(FeedbackGenerator):
    """Feedback generator using OpenAI's GPT models."""
    
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        """
        Initialize OpenAI feedback generator.
        
        Args:
            api_key: OpenAI API key (if None, will try to get from environment)
            model: GPT model to use
        """
        super().__init__()
        
        if api_key:
            openai.api_key = api_key
        else:
            openai.api_key = os.getenv('OPENAI_API_KEY')
        
        if not openai.api_key:
            logger.warning("No OpenAI API key provided. Set OPENAI_API_KEY environment variable.")
            return
        
        self.model = model
        self.is_configured = True
        logger.info(f"OpenAI feedback generator initialized with model: {model}")
    
    def generate_feedback(self, question: str, model_answer: str, 
                         student_answer: str, score: float) -> Dict:
        """Generate feedback using OpenAI GPT."""
        if not self.is_configured:
            return self._generate_fallback_feedback(question, model_answer, student_answer, score)
        
        try:
            prompt = self._create_feedback_prompt(question, model_answer, student_answer, score)
            
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert educator providing constructive feedback on student answers. Focus on being helpful, specific, and encouraging."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            feedback_text = response.choices[0].message.content.strip()
            
            # Parse the feedback into components
            feedback_components = self._parse_feedback(feedback_text)
            
            # Add automated analysis
            coverage_feedback, coverage_score = self._analyze_coverage(model_answer, student_answer)
            relevance_feedback, relevance_score = self._analyze_relevance(question, student_answer)
            grammar_feedback, grammar_score = self._analyze_grammar(student_answer)
            coherence_feedback, coherence_score = self._analyze_coherence(student_answer)
            
            return {
                'overall_feedback': feedback_text,
                'coverage': {
                    'feedback': coverage_feedback,
                    'score': coverage_score
                },
                'relevance': {
                    'feedback': relevance_feedback,
                    'score': relevance_score
                },
                'grammar': {
                    'feedback': grammar_feedback,
                    'score': grammar_score
                },
                'coherence': {
                    'feedback': coherence_feedback,
                    'score': coherence_score
                },
                'predicted_score': score,
                'generator': 'OpenAI GPT'
            }
            
        except Exception as e:
            logger.error(f"Error generating OpenAI feedback: {str(e)}")
            return self._generate_fallback_feedback(question, model_answer, student_answer, score)
    
    def _create_feedback_prompt(self, question: str, model_answer: str, 
                              student_answer: str, score: float) -> str:
        """Create a prompt for feedback generation."""
        return f"""
Please provide constructive feedback on this student's answer. Consider the question, the model answer, and the student's response.

Question: {question}

Model Answer: {model_answer}

Student Answer: {student_answer}

Predicted Score: {score}/10

Please provide feedback covering:
1. What the student did well
2. Areas for improvement
3. Specific suggestions for better answers
4. Overall assessment

Keep the feedback constructive, specific, and encouraging. Focus on helping the student learn and improve.
"""
    
    def _parse_feedback(self, feedback_text: str) -> Dict:
        """Parse the generated feedback into structured components."""
        # This is a simple parser - in practice, you might want more sophisticated parsing
        return {
            'raw_feedback': feedback_text
        }


class LocalFeedbackGenerator(FeedbackGenerator):
    """Feedback generator using local language models or rule-based approaches."""
    
    def __init__(self):
        super().__init__()
        self.is_configured = True
        logger.info("Local feedback generator initialized")
    
    def generate_feedback(self, question: str, model_answer: str, 
                         student_answer: str, score: float) -> Dict:
        """Generate feedback using local analysis."""
        try:
            # Analyze each component
            coverage_feedback, coverage_score = self._analyze_coverage(model_answer, student_answer)
            relevance_feedback, relevance_score = self._analyze_relevance(question, student_answer)
            grammar_feedback, grammar_score = self._analyze_grammar(student_answer)
            coherence_feedback, coherence_score = self._analyze_coherence(student_answer)
            
            # Generate overall feedback
            overall_feedback = self._generate_overall_feedback(
                score, coverage_score, relevance_score, grammar_score, coherence_score
            )
            
            return {
                'overall_feedback': overall_feedback,
                'coverage': {
                    'feedback': coverage_feedback,
                    'score': coverage_score
                },
                'relevance': {
                    'feedback': relevance_feedback,
                    'score': relevance_score
                },
                'grammar': {
                    'feedback': grammar_feedback,
                    'score': grammar_score
                },
                'coherence': {
                    'feedback': coherence_feedback,
                    'score': coherence_score
                },
                'predicted_score': score,
                'generator': 'Local Analysis'
            }
            
        except Exception as e:
            logger.error(f"Error generating local feedback: {str(e)}")
            return self._generate_fallback_feedback(question, model_answer, student_answer, score)
    
    def _generate_overall_feedback(self, score: float, coverage_score: float, 
                                 relevance_score: float, grammar_score: float, 
                                 coherence_score: float) -> str:
        """Generate overall feedback based on component scores."""
        feedback_parts = []
        
        # Score-based feedback
        if score >= 8:
            feedback_parts.append("Excellent work! This is a high-quality answer.")
        elif score >= 6:
            feedback_parts.append("Good work! This answer shows solid understanding.")
        elif score >= 4:
            feedback_parts.append("This answer shows some understanding but needs improvement.")
        else:
            feedback_parts.append("This answer needs significant improvement.")
        
        # Component-specific feedback
        if coverage_score < 0.5:
            feedback_parts.append("Try to cover more aspects of the topic comprehensively.")
        
        if relevance_score < 0.5:
            feedback_parts.append("Focus more directly on answering the specific question asked.")
        
        if grammar_score < 0.6:
            feedback_parts.append("Pay attention to grammar and sentence structure for better clarity.")
        
        if coherence_score < 0.6:
            feedback_parts.append("Work on organizing your ideas more clearly and using transitions.")
        
        # Positive reinforcement
        if coverage_score >= 0.7:
            feedback_parts.append("Great job covering the key points!")
        
        if relevance_score >= 0.7:
            feedback_parts.append("Excellent focus on the question!")
        
        if grammar_score >= 0.8:
            feedback_parts.append("Good grammar and writing style!")
        
        if coherence_score >= 0.8:
            feedback_parts.append("Well-organized and coherent response!")
        
        return " ".join(feedback_parts)


class FeedbackGenerator(FeedbackGenerator):
    """Fallback feedback generator for when other methods fail."""
    
    def _generate_fallback_feedback(self, question: str, model_answer: str, 
                                   student_answer: str, score: float) -> Dict:
        """Generate basic feedback when other methods fail."""
        coverage_feedback, coverage_score = self._analyze_coverage(model_answer, student_answer)
        relevance_feedback, relevance_score = self._analyze_relevance(question, student_answer)
        grammar_feedback, grammar_score = self._analyze_grammar(student_answer)
        coherence_feedback, coherence_score = self._analyze_coherence(student_answer)
        
        overall_feedback = f"Based on the analysis, this answer received a score of {score:.1f}/10. "
        
        if score >= 7:
            overall_feedback += "This is a good answer with room for minor improvements."
        elif score >= 5:
            overall_feedback += "This answer shows some understanding but needs development."
        else:
            overall_feedback += "This answer needs significant improvement in multiple areas."
        
        return {
            'overall_feedback': overall_feedback,
            'coverage': {
                'feedback': coverage_feedback,
                'score': coverage_score
            },
            'relevance': {
                'feedback': relevance_feedback,
                'score': relevance_score
            },
            'grammar': {
                'feedback': grammar_feedback,
                'score': grammar_score
            },
            'coherence': {
                'feedback': coherence_feedback,
                'score': coherence_score
            },
            'predicted_score': score,
            'generator': 'Fallback Analysis'
        }


def main():
    """Example usage of the FeedbackGenerator classes."""
    # Test with local generator
    local_generator = LocalFeedbackGenerator()
    
    question = "Explain the concept of machine learning and its applications."
    model_answer = "Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed."
    student_answer = "Machine learning is when computers learn from data automatically. It's used in many apps and websites."
    
    feedback = local_generator.generate_feedback(question, model_answer, student_answer, 7.5)
    
    print("Generated Feedback:")
    print(json.dumps(feedback, indent=2))


if __name__ == "__main__":
    main()
