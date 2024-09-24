import os
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
from src.mcqgenerator.utils import read_file, get_table_data
import streamlit as st
from langchain_community.callbacks.manager import get_openai_callback
from src.mcqgenerator.MCQGenerator import generate_evaluate_chain
from src.mcqgenerator.logger import logging

# Load environment variables
load_dotenv()

# Load response JSON
myfilepath = '/Users/gulubibleccmedia/danielklassic/artificial_intelligence/mcq_generator/Response.json'
try:
    # with open(myfilepath, 'r') as file:
    RESPONSE_JSON = {
    "1": {
        "no": "1",
        "mcq": "multiple choice questions",
        "options": {
            "a": "choice here",
            "b": "choice here",
            "c": "choice here",
            "d": "choice here"
        },
        "correct": "correct answer"
    },
    "2": {
        "no": "2",
        "mcq": "multiple choice questions",
        "options": {
            "a": "choice here",
            "b": "choice here",
            "c": "choice here",
            "d": "choice here"
        },
        "correct": "correct answer"
    },
    "3": {
        "no": "3",
        "mcq": "multiple choice questions",
        "options": {
            "a": "choice here",
            "b": "choice here",
            "c": "choice here",
            "d": "choice here"
        },
        "correct": "correct answer"
    }
}
except Exception as e:
    st.error(f"Error loading Response.json: {str(e)}")
    RESPONSE_JSON = {}

# Streamlit app
st.title("MCQs Creator Application with LangChain 🦜⛓️")

# User input form
with st.form("user_inputs"):
    uploaded_file = st.file_uploader("Upload a PDF or txt file")
    mcq_count = st.number_input("Number of MCQs", min_value=3, max_value=50, value=5)
    subject = st.text_input("Subject", max_chars=20)
    tone = st.text_input("Complexity Level of Questions", max_chars=20, placeholder="Simple")
    button = st.form_submit_button("Create MCQs")

# Process the request when the form is submitted
if button and uploaded_file is not None and subject and tone:
    with st.spinner("Generating MCQs..."):
        try:
            # Read the uploaded file
            text = read_file(uploaded_file)

            # Generate MCQs
            with get_openai_callback() as cb:
                response = generate_evaluate_chain(
                    {
                        "text": text,
                        "number": mcq_count,
                        "subject": subject,
                        "tone": tone,
                        "response_json": json.dumps(RESPONSE_JSON)
                    }
                )

            # Display token usage and cost
            st.write(f"Total Tokens: {cb.total_tokens}")
            st.write(f"Prompt Tokens: {cb.prompt_tokens}")
            st.write(f"Completion Tokens: {cb.completion_tokens}")
            st.write(f"Total Cost: ${cb.total_cost:.4f}")

            # Process and display the response
            if isinstance(response, dict):
                quiz = response.get("quiz", None)
                if quiz is not None:
                    st.subheader("Generated Quiz")
                    st.json(quiz)  # Display the raw quiz data as JSON

                    try:
                        # Ensure quiz is a dictionary
                        if isinstance(quiz, str):
                            quiz_dict = json.loads(quiz)
                        elif isinstance(quiz, dict):
                            quiz_dict = quiz
                        else:
                            raise ValueError(f"Unexpected quiz data type: {type(quiz)}")

                        # Generate table data
                        table_data = get_table_data(json.dumps(quiz_dict))
                        if table_data:
                            df = pd.DataFrame(table_data)
                            df.index = df.index + 1
                            st.subheader("Quiz Table")
                            st.table(df)
                        else:
                            st.warning("No table data generated from the quiz.")

                        # Display review if available
                        if "review" in response:
                            st.subheader("Review")
                            st.text_area(label="", value=response["review"], height=200)
                    except json.JSONDecodeError as json_error:
                        st.error(f"Error parsing JSON: {str(json_error)}")
                        st.text("Raw quiz data:")
                        st.code(quiz)  # Display the raw quiz data for debugging
                    except Exception as e:
                        st.error(f"Error processing quiz data: {str(e)}")
                        st.text("Raw quiz data:")
                        st.code(quiz)  # Display the raw quiz data for debugging
                else:
                    st.error("No quiz data found in the response")
            else:
                st.error("Unexpected response format")
                st.json(response)  # Display the entire response for debugging

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.text(traceback.format_exc())
else:
    st.info("Please fill in all fields and upload a file to generate MCQs.")

# Add some information about the app
st.sidebar.header("About")
st.sidebar.info("This app generates multiple-choice questions (MCQs) from uploaded text using AI. "
                "Upload a PDF or text file, specify the number of questions, subject, and complexity level to get started.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ by Daniel Okumu Comboni")