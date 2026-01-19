import os 
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain_core.tools import Tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

OpenweatherAPI_key = os.getenv("Openweather_API_key")
Gemini_API_key = os.getenv("Gemini_API_key")
Tavily_API_key = os.getenv("Tavily_API_key")

weather = OpenWeatherMapAPIWrapper(api_key=OpenweatherAPI_key)
llm = ChatGoogleGenerativeAI(temperature=0.7, model="gemini-1.5-flash", api_key=Gemini_API_key)

tools = [
    Tool(
        name = "get_weather",
        func = weather.run,
        description = "Useful for when you need to get the current weather in a given city. The input should be a city name, e.g., 'San Francisco' or 'New York'."
    )
]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a very helpful assistant that helps users find out the current weather in any city using the get_weather tool."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools , prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


response = agent_executor.invoke({
    "input": "What's the weather like in Bengaluru?"
})

print(response["output"])