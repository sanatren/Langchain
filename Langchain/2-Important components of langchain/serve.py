from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langserve import add_routes
import os
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_KEY")

model = ChatGroq(model="Gemma2-9b-It",groq_api_key=groq_api_key)


# 1. Create prompt template
generic_temp = "Translate the following into {language}:"

prompt = ChatPromptTemplate.from_messages(
    [("system",generic_temp),("user","{text}")]
)

parser = StrOutputParser()

##create chain
chain = prompt|model|parser

##app defination
app = FastAPI(title="Langchain Server",
              version = "1.0",
              description = "API server for using langchain runnable interfaces")


## Adding routes
add_routes(

    app,
    chain,
    path="/chain"
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,host="localhost",port=8000)



##use /docs after ur local url to acces the methods to run chain
