"""
Configuration file for AI-Based Semantic Evaluation

This file contains default settings and configuration options
for the semantic evaluation system.
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
PLOTS_DIR = BASE_DIR / "plots"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, PLOTS_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)

# Model configurations
SEMANTIC_MODELS = {
    "all-MiniLM-L6-v2": {
        "name": "all-MiniLM-L6-v2",
        "description": "Fast, lightweight model (recommended)",
        "max_length": 256,
        "device": "auto"
    },
    "all-mpnet-base-v2": {
        "name": "all-mpnet-base-v2", 
        "description": "Higher accuracy, slower",
        "max_length": 384,
        "device": "auto"
    },
    "paraphrase-multilingual-MiniLM-L12-v2": {
        "name": "paraphrase-multilingual-MiniLM-L12-v2",
        "description": "Multilingual support",
        "max_length": 256,
        "device": "auto"
    }
}

SCORING_MODELS = {
    "bert-base-uncased": {
        "name": "bert-base-uncased",
        "description": "Standard BERT model (recommended)",
        "max_length": 512,
        "device": "auto"
    },
    "roberta-base": {
        "name": "roberta-base",
        "description": "Often better performance",
        "max_length": 512,
        "device": "auto"
    },
    "distilbert-base-uncased": {
        "name": "distilbert-base-uncased",
        "description": "Faster processing",
        "max_length": 512,
        "device": "auto"
    }
}

# Default settings
DEFAULT_SETTINGS = {
    "semantic_model": "all-MiniLM-L6-v2",
    "scoring_model": "bert-base-uncased",
    "feedback_generator": "Local Analysis",
    "batch_size": 8,
    "max_length": 512,
    "device": "auto",
    "score_range": (0, 10),
    "similarity_threshold": 0.5
}

# Evaluation settings
EVALUATION_SETTINGS = {
    "include_question_context": True,
    "batch_processing": True,
    "save_embeddings": False,
    "generate_plots": True,
    "export_results": True
}

# Feedback settings
FEEDBACK_SETTINGS = {
    "coverage_weight": 0.3,
    "relevance_weight": 0.3,
    "grammar_weight": 0.2,
    "coherence_weight": 0.2,
    "min_feedback_length": 50,
    "max_feedback_length": 500
}

# API settings
API_SETTINGS = {
    "openai_model": "gpt-3.5-turbo",
    "openai_max_tokens": 500,
    "openai_temperature": 0.7,
    "request_timeout": 30,
    "max_retries": 3
}

# Logging settings
LOGGING_SETTINGS = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": LOGS_DIR / "evaluation.log",
    "max_size": 10 * 1024 * 1024,  # 10MB
    "backup_count": 5
}

# Performance settings
PERFORMANCE_SETTINGS = {
    "enable_caching": True,
    "cache_size": 1000,
    "parallel_processing": True,
    "max_workers": 4,
    "memory_limit": 4 * 1024 * 1024 * 1024  # 4GB
}

# Data validation settings
VALIDATION_SETTINGS = {
    "required_columns": ["question", "model_answer", "student_answer"],
    "optional_columns": ["score"],
    "min_text_length": 10,
    "max_text_length": 10000,
    "score_range": (0, 10),
    "allowed_file_types": ["csv", "json"]
}

# UI settings
UI_SETTINGS = {
    "page_title": "AI-Based Semantic Evaluation",
    "page_icon": "🤖",
    "layout": "wide",
    "sidebar_state": "expanded",
    "theme": "light"
}

# Export settings
EXPORT_SETTINGS = {
    "csv_encoding": "utf-8",
    "json_indent": 2,
    "include_timestamps": True,
    "compress_output": False
}

# Environment variables
ENVIRONMENT_VARIABLES = {
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    "HUGGINGFACE_TOKEN": os.getenv("HUGGINGFACE_TOKEN"),
    "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES"),
    "TOKENIZERS_PARALLELISM": "false"
}

# File paths
FILE_PATHS = {
    "sample_dataset": DATA_DIR / "sample_dataset.csv",
    "evaluation_report": "evaluation_report.json",
    "model_checkpoint": MODELS_DIR / "best_model.pth",
    "embeddings_cache": MODELS_DIR / "embeddings_cache.npy"
}

# Error messages
ERROR_MESSAGES = {
    "model_not_initialized": "Models not initialized. Please initialize models in the sidebar.",
    "invalid_file_format": "Invalid file format. Please upload a CSV file.",
    "missing_columns": "Missing required columns in dataset.",
    "api_key_missing": "OpenAI API key not provided.",
    "model_loading_error": "Error loading model. Please check your internet connection.",
    "evaluation_error": "Error during evaluation. Please check your input data."
}

# Success messages
SUCCESS_MESSAGES = {
    "models_initialized": "✅ Models initialized successfully!",
    "evaluation_complete": "✅ Evaluation completed successfully!",
    "data_loaded": "✅ Dataset loaded successfully!",
    "results_saved": "✅ Results saved successfully!",
    "feedback_generated": "✅ Feedback generated successfully!"
}

# Warning messages
WARNING_MESSAGES = {
    "low_similarity": "⚠️ Low semantic similarity detected.",
    "empty_data": "⚠️ Empty or missing data detected.",
    "slow_performance": "⚠️ Performance may be slow with current settings.",
    "memory_usage": "⚠️ High memory usage detected."
}

def get_config(section: str = None):
    """
    Get configuration settings.
    
    Args:
        section: Specific configuration section to retrieve
        
    Returns:
        Dictionary containing configuration settings
    """
    config = {
        "semantic_models": SEMANTIC_MODELS,
        "scoring_models": SCORING_MODELS,
        "default_settings": DEFAULT_SETTINGS,
        "evaluation_settings": EVALUATION_SETTINGS,
        "feedback_settings": FEEDBACK_SETTINGS,
        "api_settings": API_SETTINGS,
        "logging_settings": LOGGING_SETTINGS,
        "performance_settings": PERFORMANCE_SETTINGS,
        "validation_settings": VALIDATION_SETTINGS,
        "ui_settings": UI_SETTINGS,
        "export_settings": EXPORT_SETTINGS,
        "environment_variables": ENVIRONMENT_VARIABLES,
        "file_paths": FILE_PATHS,
        "error_messages": ERROR_MESSAGES,
        "success_messages": SUCCESS_MESSAGES,
        "warning_messages": WARNING_MESSAGES
    }
    
    if section:
        return config.get(section, {})
    
    return config

def update_config(section: str, key: str, value):
    """
    Update a configuration setting.
    
    Args:
        section: Configuration section
        key: Setting key
        value: New value
    """
    if section == "semantic_models":
        SEMANTIC_MODELS[key] = value
    elif section == "scoring_models":
        SCORING_MODELS[key] = value
    elif section == "default_settings":
        DEFAULT_SETTINGS[key] = value
    # Add more sections as needed

def validate_config():
    """
    Validate configuration settings.
    
    Returns:
        List of validation errors
    """
    errors = []
    
    # Check required directories
    for directory in [DATA_DIR, MODELS_DIR, PLOTS_DIR, LOGS_DIR]:
        if not directory.exists():
            errors.append(f"Directory {directory} does not exist")
    
    # Check model configurations
    for model_name, config in SEMANTIC_MODELS.items():
        if "name" not in config:
            errors.append(f"Semantic model {model_name} missing name")
        if "max_length" not in config:
            errors.append(f"Semantic model {model_name} missing max_length")
    
    # Check scoring model configurations
    for model_name, config in SCORING_MODELS.items():
        if "name" not in config:
            errors.append(f"Scoring model {model_name} missing name")
        if "max_length" not in config:
            errors.append(f"Scoring model {model_name} missing max_length")
    
    return errors

if __name__ == "__main__":
    # Validate configuration
    errors = validate_config()
    if errors:
        print("Configuration validation errors:")
        for error in errors:
            print(f"- {error}")
    else:
        print("✅ Configuration validation passed")
    
    # Print current configuration
    print("\nCurrent configuration:")
    config = get_config()
    for section, settings in config.items():
        print(f"\n{section.upper()}:")
        for key, value in settings.items():
            print(f"  {key}: {value}")
