import streamlit as st
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain.llms import OpenAI

OPENAI_API_KEY= st.secrets['OPENAI_API_KEY'], 
PINECONE_API_ENV = st.secrets['PINECONE_API_ENV']
PINECONE_API_KEY=st.secrets['PINECONE_API_KEY']

# Load the PDF document
#loader = PyPDFLoader('Eat_That_Frog.pdf')
#data = loader.load()

# Split the document into smaller texts
#text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 2)
#texts = text_splitter.split_documents(data)

# creating embeddings
embeddings = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY )

# initialize pinecone
pinecone.init(      
 api_key=PINECONE_API_ENV,      
 environment='gcp-starter'      
)
# Create the Pinecone index
index = pinecone.Index('info-retrieval')
index_name = 'info-retrieval'

# Initialize the Pinecone document search
embeddings=OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)
docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)
# Initialize the OpenAI LLM
llm = OpenAI(temperature = 0.7, openai_api_key = OPENAI_API_KEY)

# Load the question answering chain
chain = load_qa_chain(llm,chain_type = "stuff")

def answer_question(question):
  # Search for relevant documents
  docs = docsearch.similarity_search(question, k = 3)

  # Run the question answering chain
  qa_answer = chain.run(input_documents=docs, question=question)
  dq = {'docs': docs, 'ans': qa_answer}

  return dq

# Set the Streamlit app title
st.title('Eat That Frog! Question Answering App')

# Get the user's question
question = st.text_input('Enter your question about Eat That Frog!:')

# Answer the user's question
dq = answer_question(question)

# Display the answer to the user
st.write(dq['ans'])

#Display the documents that were used to generate the answer
st.write('Documents used to generate the answer:')
for doc in dq['docs']:
 st.write(doc)

