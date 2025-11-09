"""
AI-Based Semantic Evaluation of Descriptive Answers - Streamlit Application

This is the main Streamlit application that provides a user-friendly interface
for evaluating student answers using semantic similarity and AI-based scoring.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import time
from pathlib import Path
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import our custom modules
from data_preprocessing import DataPreprocessor
from semantic_evaluator import SemanticEvaluator, AdvancedSemanticEvaluator
from bert_scorer import BERTScorer
from feedback_generator import LocalFeedbackGenerator, OpenAIFeedbackGenerator
from evaluation_metrics import EvaluationMetrics
from document_parser import extract_text_from_file

# Page configuration
st.set_page_config(
    page_title="AI-Based Semantic Evaluation",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .feedback-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }
    .success-message {
        color: #28a745;
        font-weight: bold;
    }
    .warning-message {
        color: #ffc107;
        font-weight: bold;
    }
    .error-message {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'evaluator' not in st.session_state:
    st.session_state.evaluator = None
if 'scorer' not in st.session_state:
    st.session_state.scorer = None
if 'feedback_generator' not in st.session_state:
    st.session_state.feedback_generator = None
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = None

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 AI-Based Semantic Evaluation of Descriptive Answers</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        st.subheader("Model Settings")
        semantic_model = st.selectbox(
            "Semantic Model",
            ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "paraphrase-multilingual-MiniLM-L12-v2"],
            help="Choose the Sentence-BERT model for semantic similarity"
        )
        
        scoring_model = st.selectbox(
            "Scoring Model",
            ["bert-base-uncased", "roberta-base", "distilbert-base-uncased"],
            help="Choose the BERT model for score prediction"
        )
        
        feedback_type = st.selectbox(
            "Feedback Generator",
            ["Local Analysis", "OpenAI GPT"],
            help="Choose the feedback generation method"
        )
        
        # Leniency level
        leniency = st.select_slider(
            "Leniency Level",
            options=["Strict", "Balanced", "Lenient"],
            value="Balanced",
            help="Controls tolerance in scoring. Strict lowers, Lenient raises scores slightly."
        )
        st.session_state["leniency_level"] = leniency
        
        # OpenAI API key input
        if feedback_type == "OpenAI GPT":
            openai_key = st.text_input(
                "OpenAI API Key",
                type="password",
                help="Enter your OpenAI API key for GPT-based feedback"
            )
            if openai_key:
                os.environ['OPENAI_API_KEY'] = openai_key
        
        # Initialize models button
        if st.button("🚀 Initialize Models", type="primary"):
            with st.spinner("Initializing models..."):
                try:
                    st.session_state.evaluator = SemanticEvaluator(semantic_model)
                    st.session_state.scorer = BERTScorer(scoring_model)
                    st.session_state.feedback_generator = (
                        OpenAIFeedbackGenerator() if feedback_type == "OpenAI GPT" 
                        else LocalFeedbackGenerator()
                    )
                    st.success("✅ Models initialized successfully!")
                except Exception as e:
                    st.error(f"❌ Error initializing models: {str(e)}")
    
    # Main content tabs
    tab1, tab_docs, tab2 = st.tabs([
        "📝 Single Answer Evaluation", 
        "📄 Document Upload Evaluation",
        "📊 Batch Evaluation"
    ])
    
    with tab1:
        single_answer_evaluation()
    
    with tab_docs:
        document_upload_evaluation()
    
    with tab2:
        batch_evaluation()

def _apply_leniency(similarity: float, leniency_level: str) -> float:
    """Apply a small offset to similarity based on leniency level."""
    offsets = {
        "Strict": -0.05,
        "Balanced": 0.0,
        "Lenient": 0.05,
    }
    offset = offsets.get(leniency_level or "Balanced", 0.0)
    adjusted = max(0.0, min(1.0, similarity + offset))
    return adjusted

def _is_answer_same_as_question(question: str, student_answer: str) -> bool:
    """
    Check if student answer is essentially the same as the question.
    Returns True if student just repeated the question (should get zero).
    """
    import re
    
    # Normalize both texts: lowercase, remove extra whitespace, remove punctuation
    def normalize_text(text: str) -> str:
        # Remove punctuation and convert to lowercase
        text = re.sub(r'[^\w\s]', '', text.lower())
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text.strip()
    
    question_norm = normalize_text(question)
    answer_norm = normalize_text(student_answer)
    
    # Check if they're exactly the same (after normalization)
    if question_norm == answer_norm:
        return True
    
    # Check if student answer is very similar to question (high overlap)
    # If student answer is mostly the same words as question, it's likely copied
    question_words = set(re.findall(r'\b\w+\b', question_norm))
    answer_words = set(re.findall(r'\b\w+\b', answer_norm))
    
    # Remove common stop words for comparison
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                  'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 
                  'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
                  'what', 'how', 'why', 'when', 'where', 'who', 'which', 'this', 'that'}
    
    question_keywords = question_words - stop_words
    answer_keywords = answer_words - stop_words
    
    # If answer has very few unique keywords beyond question keywords, it's likely copied
    if len(question_keywords) > 0 and len(answer_keywords) > 0:
        unique_answer_keywords = answer_keywords - question_keywords
        # If less than 20% of answer keywords are unique (not in question), likely copied
        if len(unique_answer_keywords) / len(answer_keywords) < 0.2 and len(answer_keywords) <= len(question_keywords) * 1.2:
            return True
    
    return False

def _compute_keyword_score(model_answer: str, student_answer: str, question: str = "") -> float:
    """
    Compute keyword-based matching score (0-10) by comparing important keywords
    from model answer with student answer.
    """
    import re
    
    # Check if student just copied the question - return zero if so
    if question and _is_answer_same_as_question(question, student_answer):
        return 0.0
    
    # Common stop words to exclude
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 
        'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them', 'we', 'you'
    }
    
    # Extract words from model answer (lowercase, alphanumeric only)
    model_words = set(re.findall(r'\b[a-z]+\b', model_answer.lower()))
    model_keywords = model_words - stop_words
    
    # Extract words from student answer
    student_words = set(re.findall(r'\b[a-z]+\b', student_answer.lower()))
    student_keywords = student_words - stop_words
    
    # Calculate keyword match ratio
    if len(model_keywords) == 0:
        return 0.0
    
    # Count matched keywords
    matched_keywords = model_keywords.intersection(student_keywords)
    match_ratio = len(matched_keywords) / len(model_keywords)
    
    # Convert to 0-10 scale
    keyword_score = match_ratio * 10
    
    return round(keyword_score, 2)

def document_upload_evaluation():
    """Upload PDFs/DOCX for question paper, model answer, and student answer, then evaluate."""
    st.header("📄 Document Upload Evaluation")
    
    if not st.session_state.evaluator:
        st.warning("⚠️ Please initialize models in the sidebar first.")
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Files")
        q_file = st.file_uploader(
            "Question Paper (PDF/DOCX)", type=["pdf", "docx"], key="qfile")
        m_file = st.file_uploader(
            "Sample/Model Answer (PDF/DOCX)", type=["pdf", "docx"], key="mfile")
        s_file = st.file_uploader(
            "Student Answer Sheet (PDF/DOCX)", type=["pdf", "docx"], key="sfile")
        
        ocr_hint = st.empty()
        
        if st.button("🔍 Extract & Evaluate", type="primary"):
            if not all([q_file, m_file, s_file]):
                st.error("❌ Please upload all three documents.")
            else:
                with st.spinner("Extracting text and evaluating... This may take a moment for scanned PDFs."):
                    try:
                        q_text, q_note = extract_text_from_file(q_file.read(), q_file.name)
                        m_text, m_note = extract_text_from_file(m_file.read(), m_file.name)
                        s_text, s_note = extract_text_from_file(s_file.read(), s_file.name)
                        
                        # Show any OCR notes
                        notes = [n for n in [q_note, m_note, s_note] if n]
                        if notes:
                            ocr_hint.info("\n".join(set(notes)))
                        
                        # Use question paper text as question context (best-effort)
                        question_context = q_text[:4000] if q_text else ""
                        
                        # Check if student just copied the question - return zero if so
                        if question_context and _is_answer_same_as_question(question_context, s_text):
                            similarity = 0.0
                            adjusted_similarity = 0.0
                            keyword_score = 0.0
                        else:
                            # Compute similarity
                            similarity = st.session_state.evaluator.compute_similarity(m_text, s_text)
                            adjusted_similarity = _apply_leniency(
                                similarity, st.session_state.get("leniency_level", "Balanced")
                            )
                            
                            # Compute keyword matching score
                            keyword_score = _compute_keyword_score(m_text, s_text, question_context)
                        
                        # Generate feedback
                        feedback = st.session_state.feedback_generator.generate_feedback(
                            question_context, m_text, s_text, adjusted_similarity * 10
                        )
                        
                        st.session_state.evaluation_results = {
                            'question': '(Extracted from document)',
                            'model_answer': m_text[:2000],
                            'student_answer': s_text[:2000],
                            'similarity': adjusted_similarity,
                            'keyword_score': keyword_score,
                            'feedback': feedback,
                            'extracted': {
                                'question': q_text[:2000]
                            }
                        }
                        st.success("✅ Evaluation completed!")
                    except Exception as e:
                        st.error(f"❌ Error during extraction/evaluation: {str(e)}")
    
    with col2:
        st.subheader("Results")
        if st.session_state.evaluation_results:
            results = st.session_state.evaluation_results
            score = results['similarity'] * 10
            st.metric("Final Marks", f"{score:.1f}/10")
            interpretation = st.session_state.evaluator.get_similarity_interpretation(
                results['similarity']
            )
            st.info(f"📊 {interpretation}")
            
            st.markdown('<div class="feedback-section">', unsafe_allow_html=True)
            st.subheader("📝 Detailed Feedback")
            fb = results['feedback']
            st.write("**Overall Assessment:**")
            st.write(fb['overall_feedback'])
            for component in ['coverage', 'relevance', 'grammar', 'coherence']:
                if component in fb:
                    st.write(f"**{component.title()}:**")
                    st.write(fb[component]['feedback'])
                    st.progress(fb[component]['score'])
            
            # Keyword Matching Score
            if 'keyword_score' in results:
                st.write("**Keyword Matching Score:**")
                st.metric("Score", f"{results['keyword_score']:.1f}/10", 
                         help="Score based on keyword overlap between model answer and student answer")
            st.markdown('</div>', unsafe_allow_html=True)
            
            with st.expander("Show extracted texts (preview)"):
                st.write("**Question (excerpt):**")
                st.write(results.get('extracted', {}).get('question', ''))
                st.write("**Model Answer (excerpt):**")
                st.write(results['model_answer'])
                st.write("**Student Answer (excerpt):**")
                st.write(results['student_answer'])

def single_answer_evaluation():
    """Single answer evaluation interface."""
    st.header("📝 Single Answer Evaluation")
    
    if not st.session_state.evaluator:
        st.warning("⚠️ Please initialize models in the sidebar first.")
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Input")
        
        question = st.text_area(
            "Question",
            placeholder="Enter the question here...",
            height=100
        )
        
        model_answer = st.text_area(
            "Model Answer (Teacher's Answer)",
            placeholder="Enter the model/teacher answer here...",
            height=150
        )
        
        student_answer = st.text_area(
            "Student Answer",
            placeholder="Enter the student's answer here...",
            height=150
        )
        
        if st.button("🔍 Evaluate Answer", type="primary"):
            if not all([question, model_answer, student_answer]):
                st.error("❌ Please fill in all fields.")
                return
            
            with st.spinner("Evaluating answer..."):
                try:
                    # Check if student just copied the question - return zero if so
                    if _is_answer_same_as_question(question, student_answer):
                        similarity = 0.0
                        adjusted_similarity = 0.0
                        keyword_score = 0.0
                    else:
                        # Compute semantic similarity
                        similarity = st.session_state.evaluator.compute_similarity(
                            model_answer, student_answer
                        )
                        adjusted_similarity = _apply_leniency(
                            similarity, st.session_state.get("leniency_level", "Balanced")
                        )
                        
                        # Compute keyword matching score
                        keyword_score = _compute_keyword_score(model_answer, student_answer, question)
                    
                    # Generate feedback
                    feedback = st.session_state.feedback_generator.generate_feedback(
                        question, model_answer, student_answer, adjusted_similarity * 10
                    )
                    
                    # Store results
                    st.session_state.evaluation_results = {
                        'question': question,
                        'model_answer': model_answer,
                        'student_answer': student_answer,
                        'similarity': adjusted_similarity,
                        'keyword_score': keyword_score,
                        'feedback': feedback
                    }
                    
                    st.success("✅ Evaluation completed!")
                    
                except Exception as e:
                    st.error(f"❌ Error during evaluation: {str(e)}")
    
    with col2:
        st.subheader("Results")
        
        if st.session_state.evaluation_results:
            results = st.session_state.evaluation_results
            
            # Score display
            score = results['similarity'] * 10
            st.metric("Semantic Similarity Score", f"{score:.1f}/10")
            
            # Similarity interpretation
            interpretation = st.session_state.evaluator.get_similarity_interpretation(
                results['similarity']
            )
            st.info(f"📊 {interpretation}")
            
            # Feedback display
            st.markdown('<div class="feedback-section">', unsafe_allow_html=True)
            st.subheader("📝 Detailed Feedback")
            
            feedback = results['feedback']
            
            st.write("**Overall Assessment:**")
            st.write(feedback['overall_feedback'])
            
            # Component feedback
            components = ['coverage', 'relevance', 'grammar', 'coherence']
            for component in components:
                if component in feedback:
                    st.write(f"**{component.title()}:**")
                    st.write(feedback[component]['feedback'])
                    st.progress(feedback[component]['score'])
            
            # Keyword Matching Score
            if 'keyword_score' in results:
                st.write("**Keyword Matching Score:**")
                st.metric("Score", f"{results['keyword_score']:.1f}/10", 
                         help="Score based on keyword overlap between model answer and student answer")
            st.markdown('</div>', unsafe_allow_html=True)

def batch_evaluation():
    """Batch evaluation interface."""
    st.header("📊 Batch Evaluation")
    
    if not st.session_state.evaluator:
        st.warning("⚠️ Please initialize models in the sidebar first.")
        return
    
    st.subheader("Upload Dataset")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file with columns: question, model_answer, student_answer, score"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Dataset loaded successfully! ({len(df)} rows)")
            
            # Display dataset preview
            st.subheader("Dataset Preview")
            st.dataframe(df.head())
            
            # Validate columns
            required_columns = ['question', 'model_answer', 'student_answer']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"❌ Missing required columns: {missing_columns}")
                return
            
            if st.button("🚀 Run Batch Evaluation", type="primary"):
                with st.spinner("Running batch evaluation..."):
                    try:
                        # Run evaluation
                        results = st.session_state.evaluator.evaluate_dataset(
                            df['question'].tolist(),
                            df['model_answer'].tolist(),
                            df['student_answer'].tolist()
                        )
                        
                        # Display results
                        st.subheader("Evaluation Results")
                        
                        # Statistics
                        stats = results['statistics']
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Mean Similarity", f"{stats['mean']:.3f}")
                        with col2:
                            st.metric("Std Deviation", f"{stats['std']:.3f}")
                        with col3:
                            st.metric("Min Similarity", f"{stats['min']:.3f}")
                        with col4:
                            st.metric("Max Similarity", f"{stats['max']:.3f}")
                        
                        # Results table
                        results_df = pd.DataFrame(results['detailed_results'])
                        st.dataframe(results_df)
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name="evaluation_results.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Error during batch evaluation: {str(e)}")
                        
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")


def footer():
    """Application footer."""
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🤖 AI-Based Semantic Evaluation of Descriptive Answers</p>
        <p>Built with Streamlit, Sentence-BERT, and BERT</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    footer()
