'''
Code for rag chat with history.

import os 
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# ------ Configuration ------
api_key2 = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "rag_chat_history")

# ------ Data Pipeline ------
# Load documents
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://python.langchain.com/api_reference/core/prompts/langchain_core.prompts.chat.ChatPromptTemplate.html")
docs = loader.load()

# Split documents
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(docs)

# Create vector store
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
embeddings = OpenAIEmbeddings(api_key=api_key2, model="text-embedding-3-large")
vectorstore = FAISS.from_documents(documents, embeddings)

# ------ Conversation Setup ------
# Initialize LLM
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(api_key=api_key2, model="gpt-4-turbo-preview")

# Create prompt with history support
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
prompt = ChatPromptTemplate.from_messages([
    ("system", """You're a helpful AI assistant. Answer questions using the provided context and chat history. 
     If you don't know the answer, say so. Keep answers concise and relevant.
     
     Context: {context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

# Build chains
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = vectorstore.as_retriever()
rag_chain = create_retrieval_chain(retriever, document_chain)

# ------ Conversation Loop ------
chat_history = []

def chat(query: str, history: list) -> str:
    """Process a query with conversation history"""
    response = rag_chain.invoke({
        "input": query,
        "chat_history": history
    })
    
    # Update history with both question and answer
    history.extend([
        HumanMessage(content=query),
        AIMessage(content=response["answer"])
    ])
    
    return response["answer"]

# Example conversation
if __name__ == "__main__":
    print("Chat with RAG Assistant (type 'exit' to quit)")
    while True:
        query = input("\nYou: ")
        if query.lower() in ['exit', 'quit']:
            break
            
        response = chat(query, chat_history)
        print(f"\nAssistant: {response}")
        
    print("\nChat History:")
    for msg in chat_history:
        print(f"{msg.type}: {msg.content}")
        '''
