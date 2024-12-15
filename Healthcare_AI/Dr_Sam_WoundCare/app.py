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

st.title("Hi, I am Dr. Sam, Wound Care Specialist")
st.subheader("Please upload an image of your wound for analysis")
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
        if st.session_state['result'] == "Normal":
            st.session_state['responses'].append(
                "Hello, I am Dr. Sam! A normal wound has been detected. This type of wound is typically minor, but if you have any concerns or need advice on how to care for it, don't hesitate to ask!"
            )
        elif st.session_state['result'] == "Pressure Wounds":
            st.session_state['responses'].append(
                "Hello, I am Dr. Sam! A pressure wound has been detected. These wounds are caused by prolonged pressure on the skin and may require special care. If you need guidance on how to manage it or have any concerns, feel free to ask!"
            )
        elif st.session_state['result'] == "Surgical Wounds":
            st.session_state['responses'].append(
                "Hello, I am Dr. Sam! A surgical wound has been detected. This type of wound is the result of an incision made during surgery and may require proper cleaning and care to heal. If you need help with the healing process or have any questions, don't hesitate to ask!"
            )
        elif st.session_state['result'] == "Trauma Wounds":
            st.session_state['responses'].append(
                "Hello, I am Dr. Sam! A trauma wound has been detected. These wounds are typically caused by accidents or injuries and may need specific care to avoid complications. If you need assistance with treatment or have any questions, feel free to ask!"
            )
        elif st.session_state['result'] == "Venous Wounds":
            st.session_state['responses'].append(
                "Hello, I am Dr. Sam! A venous wound has been detected. These wounds are often related to poor circulation and may require specialized care to heal properly. If you need advice or have any questions about managing it, feel free to reach out!"
            )
            

# Check if the result is available to display chat
if st.session_state['result'] is not None:
    # Initialize the language model
    llm = ChatGroq(groq_api_key=api_key1, model_name="llama3-8b-8192", temperature=0.6)

    # Define prompt templates
    system_msg_template = SystemMessagePromptTemplate.from_template(
        template=f"You are Dr. Sam, Wound Care Specialist. After examining an image of the wound, it has been found that there is {st.session_state['result']}. Please treat the patient accordingly and offer any further guidance as needed."
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
    st.info("Please upload an image of your wound for analysis")
