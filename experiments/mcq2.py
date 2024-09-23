import os
import json
import traceback
import argparse
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain_community.callbacks.manager import get_openai_callback
import pandas as pd
import PyPDF2

load_dotenv()

KEY = os.getenv('OPEN_API_KEY')

RESPONSE_JSON = {
    "1": {
        "mcq": "multiple choice question",
        "options": {
            "a": "choice here",
            "b": "choice here",
            "c": "choice here",
            "d": "choice here"
        },
        "correct": "correct answer"
    },
    # ... (keep the rest of the RESPONSE_JSON structure)
}

def read_file(file_path):
    _, file_extension = os.path.splitext(file_path)
    if file_extension.lower() == '.pdf':
        return read_pdf(file_path)
    else:
        return read_text(file_path)

def read_text(file_path):
    with open(file_path, "r") as file:
        return file.read()

def read_pdf(file_path):
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

def generate_quiz(text, number, subject, tone):
    llm = ChatOpenAI(openai_api_key=KEY, model_name='gpt-3.5-turbo', temperature=0.5)

    TEMPLATE = """
    Text:{text}
    You are an expert MCQ maker. Given the above text, it is your job to \
    create a quiz of {number} multiple choice questions for {subject} students in {tone} tone. 
    Make sure the questions are not repeated and check all the questions to be conforming the text as well.
    Make sure to format your response like RESPONSE_JSON below and use it as a guide. \
    Ensure to make {number} MCQs
    ### RESPONSE_JSON
    {response_json}
    """

    quiz_generate_prompt = PromptTemplate(
        input_variables=["text", "number", "subject", "tone", "response_json"],
        template=TEMPLATE
    )

    quiz_chain = LLMChain(llm=llm, prompt=quiz_generate_prompt, output_key="quiz", verbose=True)

    TEMPLATE2 = """
    You are an expert English grammarian and writer. Given a Multiple Choice Quiz for {subject} students.\
    You need to evaluate the complexity of the question and give a complete analysis of the quiz if the students
    will be able to understand the questions and answer them. Only use at max 50 words for complexity analysis. 
    If the quiz is not at par with the cognitive and analytical abilities of the students,\
    update the quiz questions which need to be changed and change the tone such that it perfectly fits the student abilities
    Quiz_MCQs:
    {quiz}

    Check from an expert English Writer of the above quiz:
    """

    quiz_evaluation_prompt = PromptTemplate(input_variables=["subject", "quiz"], template=TEMPLATE2)
    review_chain = LLMChain(llm=llm, prompt=quiz_evaluation_prompt, output_key="review", verbose=True)

    generate_evaluate_chain = SequentialChain(
        chains=[quiz_chain, review_chain],
        input_variables=["text", "number", "subject", "tone", "response_json"],
        output_variables=["quiz", "review"],
        verbose=True,
    )

    with get_openai_callback() as cb:
        response = generate_evaluate_chain(
            {
                "text": text,
                "number": number,
                "subject": subject,
                "tone": tone,
                "response_json": json.dumps(RESPONSE_JSON)
            }
        )

    print(f"Total Tokens: {cb.total_tokens}")
    print(f"Prompt Tokens: {cb.prompt_tokens}")
    print(f"Completion Tokens: {cb.completion_tokens}")
    print(f"Total Cost: {cb.total_cost}")

    return response

def process_quiz(quiz_json):
    quiz = json.loads(quiz_json)
    quiz_table = []

    for key, value in quiz.items():
        mcq = value.get('mcq')
        options = " | ".join(
            [f"{option}: {option_value}" for option, option_value in value.get('options').items()]
        )
        correct = value['correct']
        quiz_table.append({"MCQ": mcq, "Choices": options, "Correct": correct})

    return pd.DataFrame(quiz_table)

def save_output(data, output_file, output_format):
    if output_format == 'csv':
        data.to_csv(output_file, index=False)
    elif output_format == 'json':
        data.to_json(output_file, orient='records', indent=2)
    print(f"Output saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate MCQs from a text or PDF file.")
    parser.add_argument("file_path", help="Path to the input file (text or PDF)")
    parser.add_argument("--number", type=int, default=5, help="Number of MCQs to generate")
    parser.add_argument("--subject", default="General Knowledge", help="Subject of the MCQs")
    parser.add_argument("--tone", default="simple", help="Tone of the MCQs")
    parser.add_argument("--output", default="quiz_output", help="Output file name (without extension)")
    parser.add_argument("--format", choices=['csv', 'json'], default='csv', help="Output file format")

    args = parser.parse_args()

    try:
        text = read_file(args.file_path)
        response = generate_quiz(text, args.number, args.subject, args.tone)
        quiz_df = process_quiz(response['quiz'])
        output_file = f"{args.output}.{args.format}"
        save_output(quiz_df, output_file, args.format)
        print("Quiz generation completed successfully.")
        print("Review:", response['review'])
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Traceback:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()