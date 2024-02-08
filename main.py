from PyPDF2 import PdfReader
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI

import os
from dotenv import load_dotenv

# Enter your query about bitcoin here
query = "What is the proposed system for electronic transactions without relying on trust?"

# Get OpenAI api key
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

assert OPENAI_API_KEY != None, "Set OPENAI_API_KEY in .env file"

# load pdf
pdf_filename = "assets/bitcoin-whitepaper.pdf"
pdf_path = os.path.join(os.getcwd(), pdf_filename)
pdf_reader = PdfReader(pdf_path)

# parse pdf text
parsed_text = ""
for i, page in enumerate(pdf_reader.pages):
    page_text = page.extract_text()
    if page_text:
        parsed_text += page_text

# break text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=32,
    length_function=len,
)
chunks_of_text = text_splitter.split_text(parsed_text)

# implement similarity search
embeddings = OpenAIEmbeddings()
docsearch = FAISS.from_texts(chunks_of_text, embeddings)

# create chain for querying
chain = load_qa_chain(ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name='gpt-3.5-turbo'), chain_type="stuff")

# Query the pdf file
docs = docsearch.similarity_search(query)

result = chain.run(input_documents=docs, question=query)
print(result)