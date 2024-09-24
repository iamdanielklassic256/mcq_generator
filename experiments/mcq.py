import os
import json
import traceback
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain_community.callbacks.manager import get_openai_callback
import pandas as pd
import PyPDF2




load_dotenv()


KEY=os.getenv('OPEN_API_KEY')
# print(KEY)

RESPONSE_JSON = {
	"1": {
		"mcq": "mutiple choice question",
		"options": {
			"a": "choice here",
			"b": "choice here",
			"c": "choice here",
			"d": "choice here"
		},
		"correct": "correct answer"
	},
	"2": {
		"mcq": "mutiple choice question",
		"options": {
			"a": "choice here",
			"b": "choice here",
			"c": "choice here",
			"d": "choice here"
		},
		"correct": "correct answer"
	},
	"3": {
		"mcq": "mutiple choice question",
		"options": {
			"a": "choice here",
			"b": "choice here",
			"c": "choice here",
			"d": "choice here"
		},
		"correct": "correct answer"
	},
	"4": {
		"mcq": "mutiple choice question",
		"options": {
			"a": "choice here",
			"b": "choice here",
			"c": "choice here",
			"d": "choice here"
		},
		"correct": "correct answer"
	},
}


llm=ChatOpenAI(openai_api_key=KEY, model_name='gpt-3.5-turbo', temperature=0.5)


# print(llm)

TEMPLATE="""
Text:{text}
You are an expert MCQ maker. Given the above text, it is your job to \
create a quiz  of {number} multiple choice questions for {subject} students in {tone} tone. 
Make sure the questions are not repeated and check all the questions to be conforming the text as well.
Make sure to format your response like  RESPONSE_JSON below  and use it as a guide. \
Ensure to make {number} MCQs
### RESPONSE_JSON
{response_json}

"""

# promt template here
quiz_generate_prompt = PromptTemplate(
	input_variables=["text", "number", "subject", "tone", "response_json"],
    template=TEMPLATE
)

quiz_chain=LLMChain(llm=llm, prompt=quiz_generate_prompt, output_key="quiz", verbose=True)


TEMPLATE2="""
You are an expert english grammarian and writer. Given a Multiple Choice Quiz for {subject} students.\
You need to evaluate the complexity of teh question and give a complete analysis of the quiz if the students
will be able to unserstand the questions and answer them. Only use at max 50 words for complexity analysis. 
if the quiz is not at par with the cognitive and analytical abilities of the students,\
update tech quiz questions which needs to be changed  and change the tone such that it perfectly fits the student abilities
Quiz_MCQs:
{quiz}

Check from an expert English Writer of the above quiz:
"""


quiz_evaluation_prompt=PromptTemplate(input_variables=["subject", "quiz"], template=TEMPLATE2)

review_chain=LLMChain(llm=llm, prompt=quiz_evaluation_prompt, output_key="review", verbose=True)


# This is an Overall Chain where we run the two chains in Sequence
generate_evaluate_chain=SequentialChain(
    chains=[quiz_chain, review_chain], 
    input_variables=["text", "number", "subject", "tone", "response_json"],
    output_variables=["quiz", "review"], 
    verbose=True,
)

file_path="/Users/gulubibleccmedia/danielklassic/artificial_intelligence/mcq_generator/data.txt"

# print(file_path)


with open(file_path, "r") as file:
    TEXT = file.read()
    
    
    # print(TEXT)
   
# serialise a python dictionary into a json formatted string 
json.dumps(RESPONSE_JSON)


NUMBER =5
SUBJECT="Machine Learning",
TONE="simple",


# how to setup token usage tracking in langchain

with get_openai_callback()as cb:
    response=generate_evaluate_chain(
		{
			"text": TEXT,
            "number": NUMBER,
            "subject": SUBJECT,
            "tone": TONE,
            "response_json": json.dumps(RESPONSE_JSON)  # serialize the response_json dictionary into a json string.
		}
	)    
    
    
# print(f"Total Tokens: {cb.total_tokens}")
# print(f"Prompt Tokens: {cb.prompt_tokens}")
# print(f"Completion Tokens: {cb.completion_tokens}")
# print(f"Total Cost: {cb.total_cost}")


quiz = response.get('quiz')
quiz = json.loads(quiz)


quiz_table = []

for key, value in quiz.items():
    mcq = value.get('mcq')
    options = " | ".join(
		[
			f"{option}: {option_value}"
            for option, option_value in value.get('options').items()	
		]
	)
    correct = value['correct']
    quiz_table.append({"MCQ": {mcq}, "Choices": options, "Correct": correct })
    



getTabulatedData = pd.DataFrame(quiz_table)

# print(getTabulatedData)

getTabulatedData.to_csv('machinelearning2.csv', index=False)