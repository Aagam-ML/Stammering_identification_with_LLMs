from langchain_cohere.llms import Cohere #cohere is a llm not a open source but provides few free tokens
from langchain_core.prompts import ChatPromptTemplate #prompt needed to run the program
from langchain_core.output_parsers import StrOutputParser
from langchain_cohere import ChatCohere
import os

def LLM_search(BOKS):
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")
    LANGCHAIN_TRACING_V2=os.getenv("LANGCHAIN_TRACING_V2")
    LANGCHAIN_API_KEY=os.getenv("LANGCHAIN_API_KEY")
    LANGCHAIN_ENDPOINT=os.getenv("LANGCHAIN_ENDPOINT")
    LANGCHAIN_PROJECT=os.getenv("LANGCHAIN_PROJECT")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system"," |Prolongation | Block | SoundRep | WordRep | DifficultToUnderstand | Interjection | These all are stammering features. from the data i provide you have to tell me which all features are applies to the text , provide ans in excel format like this |Prolongation: 0|  0 | SoundRep : 0 | WordRep : 0 | DifficultToUnderstand : 0 | Interjection : 1 |"),
                   ("user","Question:{question}")
        ]
    )

    llm = ChatCohere()
    output_parser = StrOutputParser()
    chain = prompt|llm|output_parser

    llm_result=chain.invoke({"question":BOKS})
    return llm_result
