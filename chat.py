from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# Store the chunks in vector store
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough,RunnableParallel

from langchain.chains import create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain.memory import ChatMessageHistory

# from openai import OpenAI
import streamlit as st

### Read the Gemini Key
f = open("keys/.gemini_api_key.txt")
google_api_key = f.read()

## Create the Model
chat_model = ChatGoogleGenerativeAI(google_api_key=google_api_key, 
                                   model="gemini-1.5-pro-latest")

## Create the Model
embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=google_api_key, 
                                               model="models/embedding-001")
# Setting a Connection with the ChromaDB
db_connection = Chroma(persist_directory="./chroma_db_", embedding_function=embedding_model)

# Converting CHROMA db_connection to Retriever Object
retriever = db_connection.as_retriever(search_kwargs={"k": 5})


## Define the Chat template
chat_template = ChatPromptTemplate.from_messages([
    # System Message Prompt Template
    SystemMessage(content="""You are a Helpful AI Bot. 
    You take the context and question from user. Your answer should be based on the specific context."""),

    # Human Message Prompt Template
    HumanMessagePromptTemplate.from_template("""Answer the question based on the given context.
    Context:
    {context}
    
    Question: 
    {question}
    
    Answer: """)
])

### Define the structure of the output
output_parser = StrOutputParser()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

## Create the chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | chat_template
    | chat_model
    | output_parser
)

st.title("💬Leave No Context Behind Paper Q/A RAG System")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hi, How may I help you today?"),
    ]

# converstion
for message in st.session_state.chat_history:
    if isinstance(message,AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message,HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

## Take the user input
user_prompt = st.chat_input()

if user_prompt is not None and user_prompt != "":
    st.session_state.chat_history.append(HumanMessage(content=user_prompt))

    with st.chat_message("Human"):
        st.markdown(user_prompt)

    with st.chat_message("AI"):
        response= st.write_stream(rag_chain.stream(user_prompt))

    st.session_state.chat_history.append(AIMessage(content=response))
    