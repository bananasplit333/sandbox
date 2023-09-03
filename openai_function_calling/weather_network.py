import os
import json
import openai
import requests
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA,  ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory

#initialize memory
memory = ConversationBufferMemory()

#load API 
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key
openweather_api = os.getenv("OPENWEATHER_API_KEY")

#get the coordinates of the location
def get_coordinates(location):
    response = requests.get(f"http://api.openweathermap.org/geo/1.0/direct?q={location}&limit=5&appid={openweather_api}")
    return response

#given the coordinates, return the weather details of the location
def get_weather(location):
    data = json.loads(location)
    lon = data["longitude"]
    lat = data["latitude"]
    response = requests.get(f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={openweather_api}")
    return response.json()
    

messages = [
    {"role":"system", "content": "You will call a function for the user. You will pass a keyword to the function, and receive a json file containg information about the keyword. Please return a summary of the data."},
    {"role": "user", "content": "What is the current weather in Ottawa?"}
]

#function calling. The function is called by the Chat Completion API, and the keyword is automatically detected and passed onto the get_coordinates method.
functions = [
    {
        "name": "get_coordinates",
        "description": "Get the latitude and longitude for a location given by the user.",
        "parameters":{
            "type": "object",
            "properties": {
                "latitude": {
                    "type": "number",
                    "description": "latitude"
                }, 
                "longitude": {
                    "type": "number",
                    "description": "longitude"
                }
            },
            "required": ["latitude", "longitude"]
        },
           
    }
]

#function to call the openai api 
def complete_chat():
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages = messages,
        functions = functions,
        temperature=0
    )

    if (response):
        location_details = get_weather(response["choices"][0]["message"]["function_call"]["arguments"])
        output = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You will be given a dict of a location and its weather details. Please look over the data and summarize the weather conditions in no more than 3 sentences. In the concluding sentence, recommend an outfit for the user, based on the weather conditions. Make sure to convert weather measurements to Celsius."},
                {"role": "user", "content": str(location_details)}
                ],
            temperature=0
        )
        return output
    
#call the function
ans = complete_chat()
print(ans["choices"][0]["message"]["content"])