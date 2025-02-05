import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from apiclient import APIClient

document = PyPDFLoader("./data/Documentation.pdf").load()[0].page_content
apiclient = APIClient()
print(document)

template = """
Given the context to answer the query is based on the business rules - {document}
You have to answer the query {user_query} 
return the result alone in json format.
"""

prompt_template = PromptTemplate(template=template, input_variables=["document", "user_query"])
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=os.getenv("GOOGLE_API_KEY"))
handler = prompt_template | llm | StrOutputParser()
tools = apiclient.get_tools()
agent = AgentExecutor(agent=create_react_agent(llm, tools=tools, prompt=hub.pull("hwchase17/react")),
                      tools=tools,
                      verbose=True,
                      handle_parsing_errors=True)
while True:
    user_query=input("input your question")
    if user_query == "exit":
        exit(0)
    print(agent.invoke(input={"input": prompt_template.format_prompt(document=document, user_query=user_query)})["output"] )
