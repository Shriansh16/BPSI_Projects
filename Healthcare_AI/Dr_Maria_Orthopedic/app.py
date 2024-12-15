import os
import streamlit as st
from src.utils import decodeImage
from src.pipelines.prediction_pipeline import PredictionPipeline
from dotenv import load_dotenv
from streamlit_chat import message
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain_groq import ChatGroq

load_dotenv()

api_key1 = "your api key"

st.title("Hi, I am Dr. Maria, Orthopedic Specialist")
st.subheader("Please upload the X-ray for analysis")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Initialize session state for responses and requests
if 'responses' not in st.session_state:
    st.session_state['responses'] = []
if 'requests' not in st.session_state:
    st.session_state['requests'] = []
if 'result' not in st.session_state:
    st.session_state['result'] = None

# Clear chat history when new image is uploaded
if uploaded_file is not None and uploaded_file.name != st.session_state.get('previous_file_name', ''):
    st.session_state['responses'] = []
    st.session_state['requests'] = []
    st.session_state['result'] = None
    st.session_state['buffer_memory'] = ConversationBufferWindowMemory(k=5, return_messages=True)

# Store the uploaded file name
if uploaded_file is not None:
    st.session_state['previous_file_name'] = uploaded_file.name

# If an image is uploaded
if uploaded_file is not None:
    # Save the image
    image_path = "inputImage.jpg"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Create the classifier and predict
    classifier = PredictionPipeline(image_path)

    # Button to make prediction
    if st.button("Predict"):
        st.session_state['result'] = classifier.predict()
        st.write("Prediction Result:", st.session_state['result'])

        # Set the initial response based on the prediction
        if st.session_state['result'] == "Fractured":
            st.session_state['responses'].append(
                "Hi there, I am Dr. Maria! A fracture has been detected. If you'd like to ask anything or need further guidance, feel free to ask!"
            )
        else:
            st.session_state['responses'].append(
                "Hi there, I am Dr. Maria! No fracture was detected. However, you can share your symptoms or ask any questions, and I’ll be happy to assist you."
            )

# Check if the result is available to display chat
if st.session_state['result'] is not None:
    # Initialize the language model
    llm = ChatGroq(groq_api_key=api_key1, model_name="llama3-8b-8192", temperature=0.6)

    # Define prompt templates
    system_msg_template = SystemMessagePromptTemplate.from_template(
        template=f"You are Dr. Maria, an Orthopedic Specialist. After examining an X-ray report, it has been found that there is {st.session_state['result']}. Please treat the patient accordingly and offer any further guidance as needed."
    )
    human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

    prompt_template = ChatPromptTemplate.from_messages([
        system_msg_template, 
        MessagesPlaceholder(variable_name="history"), 
        human_msg_template
    ])

    link = 'startup_logo1.jpg'

    # Create conversation chain
    conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)

    # Container for chat history
    response_container = st.container()
    # Container for text box
    text_container = st.container()

    with text_container:
        user_query = st.chat_input("Enter your query")

        if user_query:
            with st.spinner("typing..."):
                try:
                    response = conversation.predict(input=f"Query:\n{user_query}")
                    # Append the new query and response to the session state
                    st.session_state.requests.append(user_query)
                    st.session_state.responses.append(response)
                except Exception as e:
                    st.error(f"Error: {e}")

    st.markdown(
        """
        <style>
        [data-testid="stChatMessageContent"] p {
            font-size: 1rem;
        }
        </style>
        """, unsafe_allow_html=True
    )

    # Display chat history
    with response_container:
        if st.session_state['responses']:
            for i in range(len(st.session_state['responses'])):
                with st.chat_message('Momos', avatar=link):
                    st.write(st.session_state['responses'][i])
                if i < len(st.session_state['requests']):
                    message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')
else:
    st.info("Please upload an X-ray image to get started.")
