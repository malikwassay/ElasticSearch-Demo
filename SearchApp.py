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

def extract_keywords(query):
    """
    Extract relevant keywords from natural language query using OpenAI
    """
    system_prompt = """
    You are a helper that extracts relevant keywords from natural language queries about educational programs and scholarships.
    Return only the essential keywords that would be useful for searching, separated by commas.
    Do not include any other text in your response.
    """
    
    user_prompt = f"Extract search keywords from this query: {query}"
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3,
        max_tokens=100
    )
    
    keywords = response.choices[0].message.content.strip()
    return keywords

def search_programs(input_keyword, model, max_results=10):
    vector_of_input_keyword = model.encode(input_keyword)
    
    query = {
        "field": "courseDetailVector",
        "query_vector": vector_of_input_keyword,
        "k": max_results,
        "num_candidates": 500
    }
    
    res = client1.knn_search(
        index=ProgramindexName,
        knn=query,
        source=['location', 'universityName', 'overview', 'worldRanking',
                'entryRequirements', 'scholarshipsFunding', 'courseTitle',
                'courseDetail', 'qualification', 'duration', 'nextIntake', 'entryScore',
                'courseFee', 'howToApply', 'city', 'history', 'averageStartingSalary',
                'jobPlacementRatio', 'topHiringCompanies']
    )
    
    return res["hits"]["hits"]

def search_scholarships(input_keyword, model, max_results=10):
    vector_of_input_keyword = model.encode(input_keyword)
    
    query = {
        "field": "universityNameVector",
        "query_vector": vector_of_input_keyword,
        "k": max_results,
        "num_candidates": 500
    }
    
    res = client2.knn_search(
        index=ScholarshipIndexName,
        knn=query,
        source=['universityName', 'location', 'title', 'qualification', 
                'fundingDetails', 'deadline', 'eligibleIntake', 'studyMode']
    )
    
    return res["hits"]["hits"]

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
        search_query = st.text_area(
            "Enter your question in natural language",
            placeholder="For example: 'Find me computer science programs in the UK with scholarships for international students'",
            height=100
        )
        
        # Modified radio button to remove "Both" option
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
                # Extract keywords
                keywords = extract_keywords(search_query)
                st.info(f"🎯 Searching for: {keywords}")
                
                # Simplified logic for search type
                if search_type == "Programs":
                    program_results = search_programs(
                        keywords, 
                        model,
                        max_results=max_results
                    )
                    display_program_results(program_results)
                else:  # search_type == "Scholarships"
                    scholarship_results = search_scholarships(
                        keywords,
                        model,
                        max_results=max_results
                    )
                    display_scholarship_results(scholarship_results)

if __name__ == "__main__":
    main()
