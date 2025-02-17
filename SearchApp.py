import streamlit as st
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os
import json

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

# CSS styles (your existing CSS code here)
st.markdown("""
    <style>
        /* Base font settings */
        * {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 16px;
            line-height: 1.5;
        }
        
        /* Headings */
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
            font-weight: 600 !important;
        }
        
        .main-title {
            font-size: 32px !important;
            font-weight: 700 !important;
            margin-bottom: 24px !important;
        }
        
        h3 {
            font-size: 20px !important;
            margin-bottom: 16px !important;
        }
        
        /* Metrics */
        [data-testid="stMetricValue"] {
            font-size: 18px !important;
            font-weight: 600 !important;
        }
        
        [data-testid="stMetricLabel"] {
            font-size: 14px !important;
            font-weight: 400 !important;
        }
        
        /* Search container */
        .search-container {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        
        /* Result container */
        .result-container {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        /* Expander headers */
        .streamlit-expanderHeader {
            font-size: 16px !important;
            font-weight: 600 !important;
        }
        
        /* Button text */
        .stButton button {
            font-size: 16px !important;
            font-weight: 500 !important;
        }
        
        /* Input fields */
        .stTextArea textarea {
            font-size: 16px !important;
        }
        
        /* Radio buttons */
        .stRadio label {
            font-size: 16px !important;
        }
        
        /* Slider */
        .stSlider label {
            font-size: 16px !important;
        }
        
        /* Info/warning/error messages */
        .stAlert {
            font-size: 16px !important;
        }
    </style>
""", unsafe_allow_html=True)
def analyze_search_context(keywords, field_weights, client):
    """
    Use OpenAI to analyze search context and adjust field weights intelligently
    """
    system_prompt = """
    You are a search context analyzer for an educational program search engine.
    Analyze the search query and identify the importance of different aspects (score 0-10, where 10 is highest priority).
    Output a JSON object with these fields:
    - location_importance: score for location relevance
    - university_importance: score for university name relevance
    - course_importance: score for course/program relevance
    - ranking_importance: score for university ranking relevance
    - fee_importance: score for course fee relevance
    - salary_importance: score for salary/career relevance
    - qualification_importance: score for degree qualification relevance
    - detected_location: the location mentioned in the query or null if none
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze this search query: {keywords}"}
            ],
            temperature=0
        )
        
        # Extract and parse the JSON from the response
        response_text = response.choices[0].message.content
        try:
            # Handle case where response might be already formatted as JSON
            context_analysis = json.loads(response_text)
        except json.JSONDecodeError:
            # If the response contains explanation text, try to extract JSON portion
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = response_text[start_idx:end_idx]
                context_analysis = json.loads(json_str)
            else:
                raise ValueError("Could not extract JSON from response")

        # Map importance scores to field weight adjustments
        field_importance_mapping = {
            "location": {
                "fields": ["location", "locationVector", "city"],
                "score": context_analysis["location_importance"]
            },
            "university": {
                "fields": ["universityName", "universityNameVector"],
                "score": context_analysis["university_importance"]
            },
            "course": {
                "fields": ["courseTitle", "courseTitleVector", "courseDetail", "courseDetailVector"],
                "score": context_analysis["course_importance"]
            },
            "ranking": {
                "fields": ["worldRanking"],
                "score": context_analysis["ranking_importance"]
            },
            "fee": {
                "fields": ["courseFee"],
                "score": context_analysis["fee_importance"]
            },
            "salary": {
                "fields": ["averageStartingSalary"],
                "score": context_analysis["salary_importance"]
            },
            "qualification": {
                "fields": ["qualification", "qualificationVector"],
                "score": context_analysis["qualification_importance"]
            }
        }
        
        # Adjust weights based on importance scores
        adjusted_weights = field_weights.copy()
        for importance_info in field_importance_mapping.values():
            boost_factor = 1 + (importance_info["score"] / 10)  # Convert 0-10 score to multiplier
            for field in importance_info["fields"]:
                if field in adjusted_weights:
                    adjusted_weights[field] *= boost_factor
        
        # Store detected location for later use
        detected_location = context_analysis.get("detected_location")
        if detected_location:
            st.session_state['detected_location'] = detected_location
        
        return adjusted_weights, detected_location
        
    except Exception as e:
        st.error(f"Error in context analysis: {e}")
        return field_weights, None

def normalize_location(location, client):
    """
    Normalize location abbreviations to full country names using OpenAI
    """
    system_prompt = """
    You are a helper that converts country abbreviations to their full names.
    Only respond with the full country name.
    If the input is already a full country name, return it as is.
    If the input is not a recognized country or abbreviation, return it as is.
    """
    
    user_prompt = f"Convert this location if it's an abbreviation: {location}"
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            max_tokens=50
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error normalizing location: {e}")
        return location

def extract_multiple_keywords(query):
    """
    Extract relevant keywords for multiple search queries using OpenAI
    """
    system_prompt = """
    You are a helper that extracts relevant keywords from natural language queries about educational programs and scholarships.
    If the query contains multiple distinct searches, separate them with '|||'.
    For each search, provide essential keywords separated by commas.
    Keep location abbreviations as is - they will be processed separately.
    """
    
    user_prompt = f"Extract search keywords for each distinct search from this query: {query}"
    
    try:
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
        normalized_keywords_sets = []
        
        for keyword_set in keywords_sets:
            keywords = [k.strip() for k in keyword_set.split(',')]
            normalized_keywords = []
            
            for keyword in keywords:
                # Check if keyword might be a location (simple heuristic)
                if len(keyword) <= 3 or keyword.upper() == keyword:
                    normalized_location = normalize_location(keyword, client)
                    normalized_keywords.append(normalized_location)
                else:
                    normalized_keywords.append(keyword)
            
            normalized_keywords_sets.append(', '.join(normalized_keywords))
        
        return normalized_keywords_sets
        
    except Exception as e:
        st.error(f"Error in keyword extraction: {e}")
        return [query]

def search_programs(input_keywords_list, model, client, max_results=10):
    """
    Enhanced semantic search for programs with OpenAI context analysis
    """
    all_results = {}
    base_weights = {
        "courseDetailVector": 1.0,
        "overviewVector": 0.9,
        "entryRequirementsVector": 0.7,
        "scholarshipsFundingVector": 0.6,
        "courseTitleVector": 1.0,
        "universityNameVector": 0.8,
        "locationVector": 1.2
    }
    
    vector_fields = list(base_weights.keys())
    location_results = {}
    
    try:
        # Initialize Elasticsearch clients
        client1 = Elasticsearch(
            "https://149ddf030ad64e34a068782db7c12c33.us-central1.gcp.cloud.es.io:443",
            api_key="RlhmcEVwVUJ0VDVKQ3FBOFMyUDg6MWZSRll4eWJSWFdEdlZ0cjJZX2RsQQ=="
        )
    except Exception as e:
        st.error(f"Error connecting to Elasticsearch: {e}")
        return []
    
    for keywords in input_keywords_list:
        # Get context-adjusted weights and detected location
        adjusted_weights, detected_location = analyze_search_context(keywords, base_weights, client)
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
                           'courseTitle', 'courseDetail', 'qualification', 'duration',
                           'nextIntake', 'courseFee', 'city', 'averageStartingSalary']
                )
                
                for hit in res["hits"]["hits"]:
                    uni_id = f"{hit['_source']['universityName']}_{hit['_source']['courseTitle']}"
                    score = hit["_score"] * adjusted_weights[field]
                    
                    # Location-based scoring
                    source = hit['_source']
                    source_location = str(source.get('location', '')).lower()
                    
                    # Apply location boost if location was detected
                    if detected_location and detected_location.lower() in source_location:
                        score *= 2.0
                        
                    if uni_id in all_results:
                        all_results[uni_id]["score"] += score
                    else:
                        all_results[uni_id] = {
                            "hit": hit,
                            "score": score,
                            "location": source_location
                        }
                        
                    # Track results by location if location was detected
                    if detected_location:
                        if detected_location not in location_results:
                            location_results[detected_location] = []
                        if uni_id not in location_results[detected_location]:
                            location_results[detected_location].append(uni_id)
                        
            except Exception as e:
                st.warning(f"Warning: Search failed for field {field}: {str(e)}")
                continue
    
    # Check if we found results for detected location
    if detected_location and (detected_location not in location_results or not location_results[detected_location]):
        st.warning(f"⚠️ No programs found in {detected_location}. Showing results from other locations.")
    
    sorted_results = sorted(all_results.values(), 
                          key=lambda x: x["score"], 
                          reverse=True)
    
    return [item["hit"] for item in sorted_results[:max_results]]

def search_scholarships(input_keywords_list, model, client, max_results=10):
    """
    Enhanced semantic search for scholarships with OpenAI context analysis
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
    location_results = {}
    
    try:
        # Initialize Elasticsearch client for scholarships
        client2 = Elasticsearch(
            "https://149ddf030ad64e34a068782db7c12c33.us-central1.gcp.cloud.es.io:443",
            api_key="LTNmdkVwVUJ0VDVKQ3FBOFdtUzE6UXpPSThOZXhSaFNDcm50UUNNYThoZw=="
        )
    except Exception as e:
        st.error(f"Error connecting to Elasticsearch: {e}")
        return []
    
    for keywords in input_keywords_list:
        adjusted_weights, detected_location = analyze_search_context(keywords, base_weights, client)
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
                    
                    source = hit['_source']
                    source_location = str(source.get('location', '')).lower()
                    
                    if detected_location and detected_location.lower() in source_location:
                        score *= 2.0
                        
                    if scholarship_id in all_results:
                        all_results[scholarship_id]["score"] += score
                    else:
                        all_results[scholarship_id] = {
                            "hit": hit,
                            "score": score,
                            "location": source_location
                        }
                        
                    if detected_location:
                        if detected_location not in location_results:
                            location_results[detected_location] = []
                        if scholarship_id not in location_results[detected_location]:
                            location_results[detected_location].append(scholarship_id)
                        
            except Exception as e:
                st.warning(f"Warning: Search failed for field {field}: {str(e)}")
                continue
    
    if detected_location and (detected_location not in location_results or not location_results[detected_location]):
        st.warning(f"⚠️ No scholarships found in {detected_location}. Showing results from other locations.")
    
    sorted_results = sorted(all_results.values(), 
                          key=lambda x: x["score"], 
                          reverse=True)
    
    return [item["hit"] for item in sorted_results[:max_results]]

def display_program_results(results):
    """
    Display program search results in an organized format
    """
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
    """
    Display scholarship search results in an organized format
    """
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
        st.markdown("*Try natural language queries! For example:* \n\n" +
                   "*'Find affordable computer science programs in the UK with good job prospects'* or \n" +
                   "*'Show me engineering scholarships in Australia for international students'*")
        
        search_query = st.text_area(
            "Enter your question in natural language",
            placeholder="For example: 'Find affordable computer science programs in the UK with good job prospects'",
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
        
        # Add advanced filters if needed
        with st.expander("Advanced Filters"):
            st.markdown("Coming soon...")
    
    if st.button("🔍 Search", type="primary"):
        if search_query:
            with st.spinner("🔄 Processing your query..."):
                # Extract multiple keyword sets
                keywords_list = extract_multiple_keywords(search_query)
                st.info(f"🎯 Searching with refined keywords: {' | '.join(keywords_list)}")
                
                try:
                    # Search based on type
                    if search_type == "Programs":
                        program_results = search_programs(
                            keywords_list, 
                            model,
                            client,
                            max_results=max_results
                        )
                        if program_results:
                            #st.success(f"Found {len(program_results)} matching programs!")
                            display_program_results(program_results)
                        else:
                            st.info("No matching programs found. Try broadening your search criteria.")
                    else:  # search_type == "Scholarships"
                        scholarship_results = search_scholarships(
                            keywords_list,
                            model,
                            client,
                            max_results=max_results
                        )
                        if scholarship_results:
                            st.success(f"Found {len(scholarship_results)} matching scholarships!")
                            display_scholarship_results(scholarship_results)
                        else:
                            st.info("No matching scholarships found. Try broadening your search criteria.")
                
                except Exception as e:
                    st.error(f"An error occurred during search: {str(e)}")
                    st.info("Please try again with a different search query.")
        else:
            st.warning("Please enter a search query.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        st.info("Please refresh the page and try again.")
