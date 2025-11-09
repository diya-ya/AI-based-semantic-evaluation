# User Guide

## Getting Started

### System Requirements
- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- 2GB free disk space
- Internet connection for model downloads

### Installation Steps

1. **Download the application**
   - Clone the repository or download the ZIP file
   - Extract to your desired location

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python demo.py
   ```

## Using the Application

### Launching the App

1. Open terminal/command prompt
2. Navigate to the project directory
3. Run: `streamlit run app.py`
4. Open your browser to `http://localhost:8501`

### First-Time Setup

1. **Initialize Models**
   - In the sidebar, select your preferred models
   - Click "🚀 Initialize Models"
   - Wait for initialization (may take a few minutes)

2. **Configure Settings**
   - Choose semantic model (recommended: all-MiniLM-L6-v2)
   - Select scoring model (recommended: bert-base-uncased)
   - Pick feedback generator (Local Analysis for free use)

## Single Answer Evaluation

### Step-by-Step Process

1. **Navigate to Single Answer Evaluation tab**

2. **Enter Input Data**
   - **Question**: The academic question being asked
   - **Model Answer**: The teacher's or reference answer
   - **Student Answer**: The answer to be evaluated

3. **Run Evaluation**
   - Click "🔍 Evaluate Answer"
   - Wait for processing (usually 2-5 seconds)

4. **Review Results**
   - **Semantic Similarity Score**: 0-10 scale
   - **Interpretation**: Human-readable assessment
   - **Detailed Feedback**: Coverage, relevance, grammar, coherence

### Understanding the Results

**Semantic Similarity Score**
- 8-10: Excellent semantic match
- 6-8: Good semantic alignment
- 4-6: Moderate similarity
- 2-4: Low similarity
- 0-2: Poor semantic match

**Feedback Components**
- **Coverage**: How well the answer addresses the topic
- **Relevance**: How directly it answers the question
- **Grammar**: Language quality and structure
- **Coherence**: Logical flow and organization

## Batch Evaluation

### Preparing Your Data

Create a CSV file with these columns:
- `question`: The question text
- `model_answer`: The reference answer
- `student_answer`: The answer to evaluate
- `score`: (Optional) Ground truth scores

### Running Batch Evaluation

1. **Upload Dataset**
   - Go to "Batch Evaluation" tab
   - Click "Choose a CSV file"
   - Select your prepared CSV file

2. **Review Dataset**
   - Check the preview table
   - Verify column names are correct
   - Note any missing data warnings

3. **Run Evaluation**
   - Click "🚀 Run Batch Evaluation"
   - Wait for processing (time depends on dataset size)

4. **Download Results**
   - Review the results table
   - Click "📥 Download Results" to save as CSV

### Batch Results Format

The output CSV includes:
- Original data columns
- `similarity`: Semantic similarity score (0-1)
- `similarity_score_10`: Score scaled to 0-10
- `interpretation`: Human-readable similarity assessment

## Dataset Management

### Creating Sample Data

1. **Generate Sample Dataset**
   - Go to "Dataset Management" tab
   - Click "📝 Generate Sample Dataset"
   - Download the created CSV file

2. **Use Sample Data**
   - The sample includes 10 question-answer pairs
   - Covers various academic topics
   - Includes ground truth scores

### Data Format Requirements

**Required Columns:**
- `question`: String, the academic question
- `model_answer`: String, the reference answer
- `student_answer`: String, the answer to evaluate

**Optional Columns:**
- `score`: Float, ground truth scores (0-10)

**Data Quality Tips:**
- Ensure answers are complete sentences
- Avoid very short answers (< 10 words)
- Include proper punctuation
- Use consistent formatting

## Model Configuration

### Semantic Models

**all-MiniLM-L6-v2** (Recommended)
- Fast processing
- Good accuracy
- Low memory usage
- Best for most use cases

**all-mpnet-base-v2**
- Higher accuracy
- Slower processing
- More memory usage
- Use for critical evaluations

**paraphrase-multilingual-MiniLM-L12-v2**
- Multilingual support
- Good for non-English content
- Moderate performance

### Scoring Models

**bert-base-uncased** (Recommended)
- Standard BERT model
- Good balance of speed and accuracy
- Well-tested and reliable

**roberta-base**
- Often better performance
- Slower than BERT
- Use for highest accuracy needs

**distilbert-base-uncased**
- Faster processing
- Lower memory usage
- Slightly lower accuracy
- Good for real-time evaluation

### Feedback Generators

**Local Analysis**
- No API key required
- Rule-based feedback
- Consistent results
- Good for basic evaluation

**OpenAI GPT**
- AI-generated feedback
- More natural language
- Requires API key
- Better for detailed feedback

## Troubleshooting

### Common Issues

**"Please initialize models in the sidebar first"**
- Solution: Go to sidebar and click "🚀 Initialize Models"
- Wait for initialization to complete

**"Error loading file"**
- Check file format (must be CSV)
- Verify column names match requirements
- Ensure file is not corrupted

**"No data available for evaluation"**
- Check that all required columns are present
- Verify data is not empty
- Ensure text fields contain actual content

**Slow Performance**
- Use smaller models (all-MiniLM-L6-v2)
- Reduce batch size
- Close other applications
- Consider using GPU if available

### Performance Tips

**For Large Datasets:**
- Process in smaller batches
- Use faster models
- Consider preprocessing data offline

**For Real-time Evaluation:**
- Use local feedback generator
- Select faster models
- Cache model results when possible

**Memory Optimization:**
- Close unused browser tabs
- Restart the application periodically
- Use smaller batch sizes

## Best Practices

### Data Preparation
- Use clear, well-formatted questions
- Provide comprehensive model answers
- Ensure student answers are complete
- Include diverse question types

### Evaluation Strategy
- Start with single answer evaluation
- Use batch processing for large datasets
- Compare results with human evaluation
- Adjust models based on your domain

### Quality Assurance
- Validate results with sample evaluations
- Check for consistent scoring patterns
- Review feedback quality regularly
- Monitor system performance

### Integration
- Export results for further analysis
- Use API for programmatic access
- Integrate with existing systems
- Maintain evaluation logs

## Advanced Features

### Custom Model Training
- Prepare training dataset with human scores
- Use "Model Training" tab (coming soon)
- Fine-tune models for your domain
- Validate performance improvements

### Analytics Dashboard
- View performance metrics
- Analyze evaluation trends
- Compare model performance
- Export analytics data

### API Integration
- Use individual modules programmatically
- Integrate with existing systems
- Automate evaluation workflows
- Customize evaluation parameters

## Support and Resources

### Getting Help
- Check this user guide
- Review the README.md file
- Run the demo script for examples
- Contact support for issues

### Additional Resources
- API Documentation
- Technical Documentation
- Sample Datasets
- Community Forums

### Updates and Maintenance
- Check for regular updates
- Update dependencies as needed
- Backup your data regularly
- Monitor system performance
