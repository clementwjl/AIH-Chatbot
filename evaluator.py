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
from tqdm import tqdm

from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from ragas.metrics.critique import harmfulness
from ragas import evaluate

# Load your .env file

load_dotenv('./.env')

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
LANGCHAIN_API_KEY = os.getenv('LANGSMITH_API_KEY')

# Replace documents in ./docs/ with your own
loader = PyPDFDirectoryLoader("./docs/")
base_docs = loader.load()

# Define the embeddings model
embeddings_model = OpenAIEmbeddings()

# Perform embeddings and store the vectors

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    length_function=len
)

docs = text_splitter.split_documents(base_docs)

texts = docs

# Vector store
vectorstore = FAISS.from_documents(docs, embeddings_model)

# In the next section, we use LangChain to create questions based on our contexts, and then answer those questions. The output is a dataset of 5 entries of questions, answers, groundtruths and contexts, in order to evaluate our pipeline with Ragas.
# This code might take awhile to execute! 

question_schema = ResponseSchema(
name="question",
description="a question about the context."
)

question_response_schemas = [
    question_schema,
]

question_output_parser = StructuredOutputParser.from_response_schemas(question_response_schemas)
format_instructions = question_output_parser.get_format_instructions()

question_generation_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

qa_template = """\
You are a social services expert providing questions for a group of volunteers. For each context, create a question that is specific to the context. Avoid creating generic or general questions.

question: a question about the context.

Format the output as JSON with the following keys:
question

context: {context}
"""

prompt_template = ChatPromptTemplate.from_template(template=qa_template)

messages = prompt_template.format_messages(
    context=texts[0],
    format_instructions=format_instructions
)

response = question_generation_model(messages)
output_dict = question_output_parser.parse(response.content)

for k, v in output_dict.items():
    print(k)
    print(v)

qac_triples = []

for text in tqdm(texts[:5]):
    messages = prompt_template.format_messages(
        context=text,
        format_instructions=format_instructions
    )
    response = question_generation_model(messages)
    try:
        output_dict = question_output_parser.parse(response.content)
    except Exception as e:
        continue
    output_dict["context"] = text
    qac_triples.append(output_dict)


primary_ground_truth_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

answer_schema = ResponseSchema(
    name="answer",
    description="an answer to the question"
)

answer_response_schemas = [
    answer_schema,
]

answer_output_parser = StructuredOutputParser.from_response_schemas(answer_response_schemas)
format_instructions = answer_output_parser.get_format_instructions()

qa_template = """\
You are a social services expert providing questions for a group of volunteers. For each question and context, create an answer.

answer: a question about the context.

Format the output as JSON with the following keys:
answer

question: {question}
context: {context}
"""

prompt_template = ChatPromptTemplate.from_template(template=qa_template)

messages = prompt_template.format_messages(
    context=qac_triples[0]["context"],
    question=qac_triples[0]["question"],
    format_instructions=format_instructions
)

response = primary_ground_truth_llm(messages)
output_dict = answer_output_parser.parse(response.content)

for k, v in output_dict.items():
    print(k)
    print(v)

for triple in tqdm(qac_triples):
    messages = prompt_template.format_messages(
        context=triple["context"],
        question=triple["question"],
        format_instructions=format_instructions
    )
    response = primary_ground_truth_llm(messages)
    try:
        output_dict = answer_output_parser.parse(response.content)
    except Exception as e:
        continue
    triple["answer"] = output_dict["answer"]

ground_truth_qac_set = pd.DataFrame(qac_triples)
ground_truth_qac_set["context"] = ground_truth_qac_set["context"].map(lambda x: str(x.page_content))
ground_truth_qac_set = ground_truth_qac_set.rename(columns={"answer" : "ground_truth"})

eval_dataset = Dataset.from_pandas(ground_truth_qac_set)

# At this point, you have your CSV to evaluate using Ragas.
eval_dataset.to_csv("groundtruth_eval_dataset.csv")

#Evaluating the RAG Pipeline; create the evaluation functions

def create_ragas_dataset(rag_pipeline, eval_dataset):
  rag_dataset = []
  for row in tqdm(eval_dataset):
    answer = rag_pipeline({"query" : row["question"]})
    rag_dataset.append(
        {"question" : row["question"],
         "answer" : answer["result"],
         "contexts" : [context.page_content for context in answer["source_documents"]],
         "ground_truths" : [row["ground_truth"]]
         }
    )
  rag_df = pd.DataFrame(rag_dataset)
  rag_eval_dataset = Dataset.from_pandas(rag_df)
  return rag_eval_dataset

def evaluate_ragas_dataset(ragas_dataset):
  result = evaluate(
    ragas_dataset,
    metrics=[
        context_precision,
        faithfulness,
        answer_relevancy,
        context_recall,
    ],
  )
  return result

  
# Initiate the qa chain

primary_qa_llm = primary_ground_truth_llm

qa_chain = RetrievalQA.from_chain_type(
    primary_qa_llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

# Evaluate rag pipelines
basic_qa_ragas_dataset = create_ragas_dataset(qa_chain, eval_dataset)

basic_qa_ragas_dataset.to_csv("basic_qa_ragas_dataset.csv")

basic_qa_result = evaluate_ragas_dataset(basic_qa_ragas_dataset)

# Read your terminal for the final Ragas scores.
print(basic_qa_result)