import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
import speech_recognition as sr
from PIL import Image

# Set page config first
st.set_page_config(
    page_title="MyTrain Chatbot",
    page_icon="🚂",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("OpenAI API key not found. Please check your .env file.")
    st.stop()

# Initialize OpenAI with the API key
llm = OpenAI(api_key=openai_api_key)

# Load CSV file and check column names (replace with your own logic)
try:
    train_data = pd.read_csv("C:/Users/User/Downloads/Train_details_22122017 (1).csv")
except FileNotFoundError:
    st.error("CSV file not found. Please check the file path.")
    st.stop()

# Function to recognize speech using the speech_recognition library
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.markdown("<span style='color:cyan;'>Listening...</span>", unsafe_allow_html=True)
        audio = recognizer.listen(source)
        st.markdown("<span style='color:cyan;'>Finished recording.</span>", unsafe_allow_html=True)
    
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "<span style='color:red;'>Sorry, I could not understand the audio.</span>"
    except sr.RequestError:
        return "<span style='color:red;'>Sorry, there was an error with the speech recognition service.</span>"

# Custom CSS for dark theme and heading styles
custom_css = """
<style>
    body {
        background-color: #f0f0f0; /* New background color */
        color: #333333; /* Text color */
    }
    .stApp {
        background-color: #f0f0f0; /* New background color */
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .stTextInput>div>div>input {
        background-color: #ffffff;
        color: #333333;
    }
    .stFileUploader {
        background-color: #ffffff;
    }
    h1 {
        color: #ff6347; 
        background-color: #e0e0e0; /* Light gray background for headings */
        padding: 10px;
        border-radius: 10px;
    }
    h2 {
        color: #32cd32; 
        background-color: #e0e0e0; /* Light gray background for headings */
        padding: 8px;
        border-radius: 8px;
    }
    h3 {
        color: #1e90ff;  
        background-color: #e0e0e0; /* Light gray background for headings */
        padding: 6px;
        border-radius: 6px;
    }
    h4, h5 { 
        color: #1e90ff;  
        background-color: #e0e0e0; /* Light gray background for headings */
        padding: 6px;
        border-radius: 6px;
    }
    .llm-response {
        color: #4b4b4b; /* Darker color for LLM responses */
        background-color: #e0e0e0; /* Light gray background for LLM responses */
        padding: 10px;
        border-radius: 5px;
    }
</style>
"""

# Inject custom CSS
st.markdown(custom_css, unsafe_allow_html=True)

# Streamlit interface
st.title("MyTrain Chatbot")
st.write("<span style='color:orange;'>Ask about train schedules in your preferred language!</span>", unsafe_allow_html=True)

# Initialize variables to store station names
if 'from_location' not in st.session_state:
    st.session_state.from_location = ""
if 'to_location' not in st.session_state:
    st.session_state.to_location = ""

# User input options
input_method = st.selectbox("Choose your input method", ["Text", "Voice"])

# Capturing input based on selected method
if input_method == "Text":
    st.session_state.from_location = st.text_input(
        "Enter your departure station:",
        placeholder="Departure Station",
    )
    st.session_state.to_location = st.text_input(
        "Enter your destination station:",
        placeholder="Destination Station",
    )

else:
    if st.button("Record Departure Station"):
        st.session_state.from_location = recognize_speech()
        st.markdown(f"<span style='color:orange;'>Departure Station: {st.session_state.from_location}</span>", unsafe_allow_html=True)
    if st.button("Record Destination Station"):
        st.session_state.to_location = recognize_speech()
        st.markdown(f"<span style='color:orange;'>Destination Station: {st.session_state.to_location}</span>", unsafe_allow_html=True)

# General query input
general_query = st.text_input("Ask Your questions(optional):")

# When the user clicks "Check Trains"
if st.button("Check Trains"):
    if st.session_state.from_location and st.session_state.to_location:
        # Use LangChain to process
        #  the query
        query = f"Find trains from {st.session_state.from_location} to {st.session_state.to_location}"
    

        # Create a prompt template
        prompt_template = PromptTemplate(
            input_variables=["query"],
            template="Query: {query}\nAnswer:",
        )
        
        # Create an LLMChain with the prompt template
        chain = LLMChain(
            llm=llm,
            prompt=prompt_template
        )
        
        # Generate the response
        response = chain.run({"query": query})
        
        st.markdown(f"<div class='llm-response'>Available Trains:</div> <div class='llm-response'>{response}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<span style='color:red;'>Please enter both departure and destination stations.</span>", unsafe_allow_html=True)

# Handle general query
if general_query:
    # Create a prompt template for general queries
    general_prompt_template = PromptTemplate(
        input_variables=["general_query"],
        template="General Query: {general_query}\nAnswer:",
    )

    # Create an LLMChain with the general query prompt template
    general_chain = LLMChain(
        llm=llm,
        prompt=general_prompt_template
    )

    # Generate the response for the general query
    general_response = general_chain.run({"general_query": general_query})

    st.markdown(f"<div class='llm-response'>Response:</div> <div class='llm-response'>{general_response}</div>", unsafe_allow_html=True)
