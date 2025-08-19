# Education Search Application

A sophisticated semantic search application for finding educational programs and scholarships using natural language queries. Built with Streamlit, Elasticsearch, and OpenAI integration for intelligent search capabilities.

## Features

### Core Functionality
- **Natural Language Search**: Query using conversational language instead of keywords
- **Dual Search Modes**: Separate search for educational programs and scholarships
- **Semantic Vector Search**: Uses sentence transformers for meaning-based search
- **Context-Aware Results**: OpenAI integration for intelligent query analysis
- **Location-Specific Filtering**: Automatic detection and filtering by geographic location
- **Smart Field Weighting**: Dynamic adjustment of search importance based on query context

### Advanced Capabilities
- **Multi-Query Processing**: Handles complex queries with multiple search intents
- **Location Normalization**: Converts abbreviations to full country names
- **Intelligent Scoring**: Context-aware result ranking with location boosting
- **Comprehensive Display**: Detailed program and scholarship information presentation

## Prerequisites

### Required Services
- **Elasticsearch Cloud**: Vector search backend
- **OpenAI API**: Natural language processing
- **Python 3.8+**: Runtime environment

### API Keys Required
- OpenAI API key
- Elasticsearch API keys (separate for programs and scholarships)

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/malikwassay/ElasticSearch-Demo
cd education-search-app
```

### 2. Install Dependencies
```bash
pip install streamlit elasticsearch sentence-transformers openai
```

Or using requirements.txt:
```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

Create a `.streamlit/secrets.toml` file:
```toml
OPENAI_API_KEY = "your-openai-api-key-here"
```

### 4. Configure Elasticsearch Credentials

Update the Elasticsearch connection details in the code:
- Replace the Elasticsearch URLs
- Update API keys for both program and scholarship indices

## Dependencies

```txt
streamlit>=1.28.0
elasticsearch>=8.0.0
sentence-transformers>=2.2.0
openai>=1.0.0
```

## Usage

### Starting the Application
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Search Examples

**Programs:**
- "Find computer science programs in the UK"
- "Show me MBA programs in Australia"
- "Engineering courses in Canada with high salary"
- "Best universities for data science in USA"

**Scholarships:**
- "Scholarships for international students in UK"
- "PhD funding opportunities in computer science"
- "Merit-based scholarships in Australia"
- "Graduate scholarships for engineering"

### Interface Components

**Search Input:**
- Natural language text area for queries
- Radio button selection for Programs vs Scholarships
- Results slider (1-50 results)

**Results Display:**
- Expandable sections for detailed information
- Metrics for key data points
- Organized layout with university, location, and program details

## Technical Architecture

### Search Pipeline

1. **Query Processing**
   - Natural language input parsing
   - Multiple keyword extraction using OpenAI
   - Location detection and normalization

2. **Context Analysis**
   - OpenAI analyzes search intent
   - Dynamic field weight adjustment
   - Importance scoring for different aspects

3. **Vector Search**
   - Sentence transformer encoding
   - Elasticsearch KNN search across multiple fields
   - Score aggregation with context weighting

4. **Result Processing**
   - Location-specific filtering
   - Score normalization and ranking
   - Deduplication and result limiting

### Data Models

**Program Index Fields:**
- Course information (title, detail, qualification)
- University data (name, ranking, location)
- Financial data (fees, starting salary)
- Academic requirements (entry score, duration)
- Career information (job placement, hiring companies)

**Scholarship Index Fields:**
- Scholarship details (title, funding, deadline)
- Eligibility criteria (qualification, intake)
- University and location information
- Study mode and application requirements

### Vector Fields

**Programs:**
- `courseDetailVector`: Course description embeddings
- `courseTitleVector`: Course name embeddings
- `universityNameVector`: University name embeddings
- `locationVector`: Geographic location embeddings
- `overviewVector`: University overview embeddings
- `entryRequirementsVector`: Admission requirements embeddings
- `scholarshipsFundingVector`: Financial information embeddings

**Scholarships:**
- `titleVector`: Scholarship title embeddings
- `universityNameVector`: University name embeddings
- `fundingDetailsVector`: Funding information embeddings
- `qualificationVector`: Eligibility requirements embeddings
- `locationVector`: Geographic location embeddings

## Configuration

### Elasticsearch Settings

**Connection Configuration:**
```python
# Programs Index
client1 = Elasticsearch(
    "your-elasticsearch-url",
    api_key="your-programs-api-key"
)

# Scholarships Index
client2 = Elasticsearch(
    "your-elasticsearch-url", 
    api_key="your-scholarships-api-key"
)
```

**Index Names:**
- Programs: `"programs"`
- Scholarships: `"scholar"`

### OpenAI Integration

**Models Used:**
- `gpt-3.5-turbo`: Query analysis and location normalization
- Temperature: 0 for consistency
- Max tokens: 50-200 depending on task

**System Prompts:**
- Context analysis for field weight adjustment
- Location normalization for geographic queries
- Keyword extraction for multi-intent queries

### Sentence Transformers

**Model:** `all-mpnet-base-v2`
- High-quality sentence embeddings
- 768-dimensional vectors
- Optimized for semantic similarity

## API Endpoints

The application doesn't expose REST APIs but uses internal functions:

### Search Functions
- `search_programs()`: Semantic program search
- `search_scholarships()`: Semantic scholarship search
- `analyze_search_context()`: OpenAI context analysis
- `extract_multiple_keywords()`: Query processing

### Utility Functions
- `normalize_location()`: Location standardization
- `display_program_results()`: Program result formatting
- `display_scholarship_results()`: Scholarship result formatting

## Customization

### Styling
The application includes comprehensive CSS styling:
- Custom fonts (Inter)
- Result containers with shadows
- Responsive metric displays
- Search container styling

### Field Weights
Adjust base weights in search functions:
```python
base_weights = {
    "courseDetailVector": 1.0,
    "courseTitleVector": 1.0,
    "universityNameVector": 0.8,
    "locationVector": 1.2,
    # Add more fields as needed
}
```

### Search Parameters
- `k`: Number of nearest neighbors (max_results * 2)
- `num_candidates`: Search space size (1000)
- Maximum results: 1-50 (user configurable)

## Error Handling

### Common Issues

**Elasticsearch Connection:**
- Verify API keys and URLs
- Check network connectivity
- Ensure indices exist

**OpenAI API:**
- Validate API key
- Monitor rate limits
- Handle JSON parsing errors

**Search Errors:**
- Graceful degradation for failed fields
- Warning messages for partial failures
- Fallback to basic search when context analysis fails

### Debugging

Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Monitor Streamlit logs:
```bash
streamlit run app.py --logger.level debug
```

## Performance Considerations

### Search Optimization
- Vector field caching in Elasticsearch
- Batch processing for multiple queries
- Smart result deduplication
- Strategic field weight adjustment

### API Usage
- OpenAI request optimization
- Sentence transformer model caching
- Elasticsearch connection pooling

### Memory Management
- Streamlit session state for detected locations
- Efficient result processing
- Limited vector dimensions

## Security

### API Key Management
- Store keys in Streamlit secrets
- Never commit keys to version control
- Use environment variables in production

### Data Privacy
- No persistent user data storage
- Session-based search state
- Secure API communications

## Deployment

### Local Development
```bash
streamlit run app.py
```

### Streamlit Cloud
1. Connect GitHub repository
2. Configure secrets in Streamlit Cloud dashboard
3. Deploy automatically from main branch

### Production Considerations
- Environment variable management
- HTTPS configuration
- Rate limiting implementation
- Error monitoring and logging

## Troubleshooting

### Search Returns No Results
- Check Elasticsearch index status
- Verify API keys and connectivity
- Review query complexity
- Test with simpler search terms

### OpenAI Context Analysis Fails
- Validate API key
- Check rate limits
- Fallback to basic field weights
- Monitor response format

### Performance Issues
- Reduce max_results parameter
- Optimize vector field selection
- Check Elasticsearch cluster health
- Monitor API response times

## Future Enhancements

### Planned Features
- Advanced filtering options
- Search result export
- User preference learning
- Multi-language support
- Saved search functionality

### Technical Improvements
- Caching layer implementation
- Advanced query optimization
- Real-time index updates
- Performance monitoring dashboard

**Note**: This application requires active Elasticsearch and OpenAI API connections. Ensure all credentials are properly configured before deployment.
