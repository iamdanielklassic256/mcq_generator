from setuptools import find_packages,setup

setup(
    name='mcqgenrator',
    version='0.0.1',
    author='Daniel Okumu Comboni',
    author_email='okumucomboni@gmail.com',
    description='A package to generate multiple choice questions',
    install_requires=["openai","langchain","streamlit","python-dotenv","PyPDF2"],
    packages=find_packages()
)