import streamlit as st
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

ProgramindexName = "programs"
ScholarshipIndexName = "scholar"

st.set_page_config(
    page_title="Education Search",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Replace the existing CSS with this refined version

st.markdown("""
    <style>
    /* Reset and base styles */
    .main-title {
        font-size: 2rem !important;
        font-weight: 600 !important;
        margin-bottom: 1.5rem !important;
        color: #2c3e50 !important;
        text-align: center;
        padding: 1.5rem 0;
        border-bottom: 2px solid #eef2f7;
    }
    
    /* Search container styling */
    .search-container {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        border: 1px solid #eef2f7;
        margin-bottom: 1.5rem;
    }
    
    /* Result container styling */
    .result-container {
        background: white;
        padding: 1.5rem;
        border-radius: 6px;
        margin-bottom: 1rem;
        border: 1px solid #eef2f7;
        transition: box-shadow 0.2s ease;
    }
    
    .result-container:hover {
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
    }
    
    /* Input fields */
    .stTextInput > div > div > input {
        font-size: 1rem !important;
        padding: 0.75rem !important;
    }
    
    .stTextArea > div > div > textarea {
        font-size: 1rem !important;
        padding: 0.75rem !important;
        min-height: 100px !important;
    }
    
    /* Button styling */
    .stButton > button {
        font-size: 1rem !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 500 !important;
        background: #2c3e50 !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
    }
    
    .stButton > button:hover {
        background: #34495e !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Radio button styling */
    .stRadio > div {
        background-color: white;
        padding: 0.75rem;
        border-radius: 6px;
        border: 1px solid #eef2f7;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-size: 1rem !important;
        font-weight: 500 !important;
        background: white !important;
        border-radius: 6px !important;
        border: 1px solid #eef2f7 !important;
    }
    
    /* Metric styling */
    .stMetric {
        background: white;
        padding: 0.75rem;
        border-radius: 6px;
        border: 1px solid #eef2f7;
    }
    
    .stMetric label {
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        color: #2c3e50 !important;
    }
    
    .stMetric .metric-value {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }
    
    /* Headers styling */
    h1, h2, h3 {
        color: #2c3e50 !important;
        font-weight: 600 !important;
    }
    
    h3 {
        font-size: 1.2rem !important;
        margin-bottom: 1rem !important;
    }
    
    /* Slider styling */
    .stSlider {
        padding: 1rem 0;
    }
    
    .stSlider > div > div > div {
        background-color: #2c3e50 !important;
    }
    
    /* Info boxes styling */
    .stAlert {
        background-color: #f8fafc !important;
        border: 1px solid #eef2f7 !important;
        padding: 0.75rem !important;
        border-radius: 6px !important;
    }
    
    /* Spinner styling */
    .stSpinner > div {
        border-color: #2c3e50 !important;
    }
    
    /* General text styling */
    p, div {
        font-size: 1rem !important;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f8fafc;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #94a3b8;
    }
    </style>
""", unsafe_allow_html=True)

try:
    # Client for programs index
    client1 = Elasticsearch(
        "https://149ddf030ad64e34a068782db7c12c33.us-central1.gcp.cloud.es.io:443",
        api_key="RlhmcEVwVUJ0VDVKQ3FBOFMyUDg6MWZSRll4eWJSWFdEdlZ0cjJZX2RsQQ=="
    )
    
    # Client for scholarships index
    client2 = Elasticsearch(
        "https://149ddf030ad64e34a068782db7c12c33.us-central1.gcp.cloud.es.io:443",
        api_key="LTNmdkVwVUJ0VDVKQ3FBOFdtUzE6UXpPSThOZXhSaFNDcm50UUNNYThoZw=="
    )
except ConnectionError as e:
    st.error(f"Connection Error: {e}")

def extract_multiple_keywords(query):
    """
    Extract relevant keywords for multiple search queries using OpenAI
    Returns a list of keyword sets
    """
    system_prompt = """
    You are a helper that extracts relevant keywords from natural language queries about educational programs and scholarships.
    If the query contains multiple distinct searches, separate them with '|||'.
    For each search, provide essential keywords separated by commas.
    Example output for multiple searches:
    computer science, UK, international students ||| data science, USA, scholarship
    """
    
    user_prompt = f"Extract search keywords for each distinct search from this query: {query}"
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3,
        max_tokens=200
    )
    
    keywords_sets = response.choices[0].message.content.strip().split('|||')
    return [kw.strip() for kw in keywords_sets]

def analyze_search_context(keywords, field_weights):
    """
    Analyze keywords to determine search context and adjust field weights
    """
    # Convert keywords to lowercase for matching
    keywords_lower = keywords.lower()
    
    # Context indicators and their corresponding fields
    context_mapping = {
        'location': ['location', 'locationVector'],
        'university': ['universityName', 'universityNameVector'],
        'course': ['courseTitle', 'courseTitleVector'],
        'ranking': ['worldRanking'],
        'fee': ['courseFee'],
        'city': ['city'],
        'salary': ['averageStartingSalary'],
        'qualification': ['qualification', 'qualificationVector'],
        'deadline': ['deadline'],
        'funding': ['fundingDetails', 'fundingDetailsVector']
    }
    
    # Keywords that indicate specific contexts
    context_keywords = {
        'location': ['in', 'at', 'country', 'location'],
        'university': ['university', 'college', 'institution'],
        'course': ['program', 'course', 'degree', 'study'],
        'ranking': ['ranking', 'ranked', 'top'],
        'fee': ['fee', 'cost', 'price', 'expensive', 'cheap'],
        'city': ['city', 'town'],
        'salary': ['salary', 'pay', 'earning'],
        'qualification': ['qualification', 'degree', 'certificate'],
        'deadline': ['deadline', 'due date', 'closing date'],
        'funding': ['funding', 'scholarship', 'financial']
    }
    
    # Identify contexts present in the search
    active_contexts = []
    for context, indicators in context_keywords.items():
        if any(indicator in keywords_lower for indicator in indicators):
            active_contexts.append(context)
    
    # Adjust weights based on identified contexts
    adjusted_weights = field_weights.copy()
    if active_contexts:
        # Boost weights for relevant fields
        boost_factor = 2.0
        for context in active_contexts:
            for field in context_mapping[context]:
                if field in adjusted_weights:
                    adjusted_weights[field] *= boost_factor
    
    return adjusted_weights

def search_programs(input_keywords_list, model, max_results=10):
    """
    Context-aware semantic search for programs
    """
    all_results = {}
    base_weights = {
        "courseDetailVector": 1.0,
        "overviewVector": 0.9,
        "entryRequirementsVector": 0.7,
        "scholarshipsFundingVector": 0.6,
        "courseTitleVector": 1.0,
        "universityNameVector": 0.8,
        "locationVector": 0.8
    }
    
    vector_fields = list(base_weights.keys())
    
    for keywords in input_keywords_list:
        # Analyze context and adjust weights
        adjusted_weights = analyze_search_context(keywords, base_weights)
        vector_of_input_keyword = model.encode(keywords)
        
        for field in vector_fields:
            query = {
                "field": field,
                "query_vector": vector_of_input_keyword,
                "k": max_results * 2,
                "num_candidates": 1000
            }
            
            try:
                res = client1.knn_search(
                    index=ProgramindexName,
                    knn=query,
                    source=['location', 'universityName', 'overview', 'worldRanking',
                           'entryRequirements', 'scholarshipsFunding', 'courseTitle',
                           'courseDetail', 'qualification', 'duration', 'nextIntake', 
                           'entryScore', 'courseFee', 'howToApply', 'city', 'history',
                           'averageStartingSalary', 'jobPlacementRatio', 'topHiringCompanies']
                )
                
                for hit in res["hits"]["hits"]:
                    uni_id = f"{hit['_source']['universityName']}_{hit['_source']['courseTitle']}"
                    score = hit["_score"] * adjusted_weights[field]
                    
                    # Additional context-based scoring
                    source = hit['_source']
                    if keywords.lower() in str(source.get('location', '')).lower():
                        score *= 1.5
                    if keywords.lower() in str(source.get('courseTitle', '')).lower():
                        score *= 1.5
                    if keywords.lower() in str(source.get('universityName', '')).lower():
                        score *= 1.5
                    
                    if uni_id in all_results:
                        all_results[uni_id]["score"] += score
                    else:
                        all_results[uni_id] = {
                            "hit": hit,
                            "score": score
                        }
                        
            except Exception as e:
                st.warning(f"Warning: Search failed for field {field}: {str(e)}")
                continue
    
    sorted_results = sorted(all_results.values(), 
                          key=lambda x: x["score"], 
                          reverse=True)
    
    return [item["hit"] for item in sorted_results[:max_results]]

def search_scholarships(input_keywords_list, model, max_results=10):
    """
    Context-aware semantic search for scholarships
    """
    all_results = {}
    base_weights = {
        "universityNameVector": 0.8,
        "titleVector": 1.0,
        "fundingDetailsVector": 0.9,
        "qualificationVector": 0.7,
        "locationVector": 0.6
    }
    
    vector_fields = list(base_weights.keys())
    
    for keywords in input_keywords_list:
        # Analyze context and adjust weights
        adjusted_weights = analyze_search_context(keywords, base_weights)
        vector_of_input_keyword = model.encode(keywords)
        
        for field in vector_fields:
            query = {
                "field": field,
                "query_vector": vector_of_input_keyword,
                "k": max_results * 2,
                "num_candidates": 1000
            }
            
            try:
                res = client2.knn_search(
                    index=ScholarshipIndexName,
                    knn=query,
                    source=['universityName', 'location', 'title', 'qualification', 
                           'fundingDetails', 'deadline', 'eligibleIntake', 'studyMode']
                )
                
                for hit in res["hits"]["hits"]:
                    scholarship_id = f"{hit['_source']['universityName']}_{hit['_source']['title']}"
                    score = hit["_score"] * adjusted_weights[field]
                    
                    # Additional context-based scoring
                    source = hit['_source']
                    if keywords.lower() in str(source.get('location', '')).lower():
                        score *= 1.5
                    if keywords.lower() in str(source.get('title', '')).lower():
                        score *= 1.5
                    if keywords.lower() in str(source.get('universityName', '')).lower():
                        score *= 1.5
                    
                    if scholarship_id in all_results:
                        all_results[scholarship_id]["score"] += score
                    else:
                        all_results[scholarship_id] = {
                            "hit": hit,
                            "score": score
                        }
                        
            except Exception as e:
                st.warning(f"Warning: Search failed for field {field}: {str(e)}")
                continue
    
    sorted_results = sorted(all_results.values(), 
                          key=lambda x: x["score"], 
                          reverse=True)
    
    return [item["hit"] for item in sorted_results[:max_results]]

def display_program_results(results):
    for result in results:
        with st.container():
            if '_source' in result:
                source = result['_source']
                
                st.markdown('<div class="result-container">', unsafe_allow_html=True)
                
                # Display university info with metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("University", source.get('universityName', 'N/A'))
                with col2:
                    st.metric("Location", f"{source.get('city', 'N/A')}, {source.get('location', 'N/A')}")
                with col3:
                    st.metric("World Ranking", source.get('worldRanking', 'N/A'))
                
                # Display career info
                with st.expander("💼 Career Information"):
                    career_col1, career_col2 = st.columns(2)
                    with career_col1:
                        # Fixed salary formatting
                        salary = source.get('averageStartingSalary')
                        if isinstance(salary, (int, float)):
                            formatted_salary = f"${salary:,}"
                        else:
                            formatted_salary = 'N/A'
                        st.metric("Average Starting Salary", formatted_salary)
                    with career_col2:
                        st.metric("Job Placement Ratio", source.get('jobPlacementRatio', 'N/A'))
                    st.write("**Top Hiring Companies:**", source.get('topHiringCompanies', 'N/A'))
                
                # Display course info
                with st.expander("📚 Course Information"):
                    st.subheader(source.get('courseTitle', 'N/A'))
                    st.write("**Overview:**", source.get('overview', 'N/A'))
                    st.write("**Course Details:**", source.get('courseDetail', 'N/A'))
                    
                    course_col1, course_col2 = st.columns(2)
                    with course_col1:
                        st.metric("Qualification", source.get('qualification', 'N/A'))
                        st.metric("Duration", source.get('duration', 'N/A'))
                    with course_col2:
                        st.metric("Next Intake", source.get('nextIntake', 'N/A'))
                
                # Display requirements and fees
                with st.expander("📋 Requirements & Fees"):
                    st.write("**Entry Requirements:**", source.get('entryRequirements', 'N/A'))
                    
                    req_col1, req_col2 = st.columns(2)
                    with req_col1:
                        st.metric("Entry Score", source.get('entryScore', 'N/A'))
                    with req_col2:
                        # Format course fee if it's a number
                        fee = source.get('courseFee')
                        if isinstance(fee, (int, float)):
                            formatted_fee = f"${fee:,}"
                        else:
                            formatted_fee = 'N/A'
                        st.metric("Course Fee", formatted_fee)
                    
                    st.write("**How To Apply:**", source.get('howToApply', 'N/A'))
                
                st.markdown('</div>', unsafe_allow_html=True)

def display_scholarship_results(results):
    for result in results:
        with st.container():
            if '_source' in result:
                source = result['_source']
                
                st.markdown('<div class="result-container">', unsafe_allow_html=True)
                
                # Display scholarship header with metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Scholarship", source.get('title', 'N/A'))
                with col2:
                    st.metric("University", f"{source.get('universityName', 'N/A')} - {source.get('location', 'N/A')}")
                
                # Display scholarship details
                with st.expander("🎓 Scholarship Details"):
                    st.write("**Funding Details:**", source.get('fundingDetails', 'N/A'))
                    
                    detail_col1, detail_col2 = st.columns(2)
                    with detail_col1:
                        st.metric("Qualification", source.get('qualification', 'N/A'))
                        st.metric("Study Mode", source.get('studyMode', 'N/A'))
                    with detail_col2:
                        st.metric("Eligible Intake", source.get('eligibleIntake', 'N/A'))
                        st.metric("Deadline", source.get('deadline', 'N/A'))
                
                st.markdown('</div>', unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-title">🎓 Education Program & Scholarship Search</h1>', unsafe_allow_html=True)
    
    # Initialize the model
    model = SentenceTransformer('all-mpnet-base-v2')
    
    # Create two columns for the layout
    search_col, filter_col = st.columns([2, 1])
    
    with search_col:
        st.markdown('<div class="search-container">', unsafe_allow_html=True)
        st.markdown("### 🔍 Ask me anything about programs or scholarships!")
        st.markdown("*You can search for multiple programs/scholarships at once! For example:* \n\n" +
                   "*'Find computer science programs in the UK and data science programs in the USA'*")
        search_query = st.text_area(
            "Enter your question in natural language",
            placeholder="For example: 'Find computer science programs in the UK and data science programs in the USA'",
            height=100
        )
        
        search_type = st.radio(
            "What are you looking for?",
            ["Programs", "Scholarships"],
            horizontal=True
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with filter_col:
        st.markdown("### ⚙️ Search Filters")
        max_results = st.slider("Maximum Results", 1, 50, 10)
    
    if st.button("🔍 Search", type="primary"):
        if search_query:
            with st.spinner("🔄 Processing your query..."):
                # Extract multiple keyword sets
                keywords_list = extract_multiple_keywords(search_query)
                st.info(f"🎯 Searching for multiple queries: {' | '.join(keywords_list)}")
                
                # Search based on type
                if search_type == "Programs":
                    program_results = search_programs(
                        keywords_list, 
                        model,
                        max_results=max_results
                    )
                    display_program_results(program_results)
                else:  # search_type == "Scholarships"
                    scholarship_results = search_scholarships(
                        keywords_list,
                        model,
                        max_results=max_results
                    )
                    display_scholarship_results(scholarship_results)

if __name__ == "__main__":
    main()
