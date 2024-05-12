import streamlit as st
import requests

# Define the URL of your FastAPI server
url = "http://localhost:8001/chat"

# Use Streamlit to get user input
user_input = st.text_input("Enter some text")

# When the user presses the button, make a request to your FastAPI server
if st.button("Submit"):
    response = requests.post(url, data={"userMessage": user_input})
    st.write(response.json())
