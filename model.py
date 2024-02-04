import os
import openai
import sys
import numpy as np
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain

def getResponse(question: str) -> str:

    load_dotenv('./.env')

    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    LANGCHAIN_API_KEY = os.getenv('LANGSMITH_API_KEY')

    loader = PyPDFDirectoryLoader("./docs/")
    base_docs = loader.load()
    print(len(base_docs))

    # embeddings_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    embeddings_model = OpenAIEmbeddings()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )

    docs = text_splitter.split_documents(base_docs)

    texts = docs

    vectorstore = FAISS.from_documents(docs, embeddings_model)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key='question',
        output_key='answer'
    )

    # Code below will enable tracing so we can take a deeper look into the chain
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_PROJECT"] = "Chatbot"


    # Define parameters for retrival
    retriever=vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": .5, "k": 5})

    # Define llm model

    llm_name = "gpt-3.5-turbo-1106"
    llm = ChatOpenAI(model_name=llm_name, temperature=0)

    # Define template prompt
    template = """You are a social services expert providing questions for a group of volunteers. For each question and context, create an answer.
    Context: {context}
    Question: {question}
    Helpful Answer:"""

    your_prompt = PromptTemplate.from_template(template)

    # Execute chain
    qa = ConversationalRetrievalChain.from_llm(
        llm,
        combine_docs_chain_kwargs={"prompt": your_prompt},
        retriever=retriever,
        return_source_documents=True,
        return_generated_question=True,
        memory=memory
    )

    # Evaluate your chatbot with questions
    result = qa({"question": question})
    print(result)
    return result['answer']
