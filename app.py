# Import necessary libraries
import streamlit as st
import uuid
import os
import psycopg
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_postgres import PostgresChatMessageHistory

# Load OpenAI API Key
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize the model
model = ChatOpenAI(model = "gpt-3.5-turbo")

# Initialize the prompt template
human_template = f"{{question}}"
prompt_template = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="history"),
        ("human", human_template),
    ]
)

# Chain the prompt and model
chain = prompt_template | model

# Define table name for PostgreSQL
table_name = "chat_history"

# Get chat history from PostgreSQL given session ID
def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
    sync_connection = psycopg.connect(
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        host="localhost",
        port="5432",
    )
    return PostgresChatMessageHistory(
        table_name, session_id, sync_connection=sync_connection
    )

# Chain with history
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_by_session_id,
    input_messages_key="question",
    history_messages_key="history",
)

# Delete chat history from PostgreSQL using session ID
def delete_chat_history(session_id: str):
    with psycopg.connect(
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        host="localhost",
        port="5432",
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {table_name} WHERE session_id = %s", (session_id,)
            )
            conn.commit()

# Initialize session state variables
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())  # Generate unique session ID for a user

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Initialize chat history

# Title of the website
st.title("Local Chatbot using ChatOpenAI")

# Display chat history
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.chat_message("user").markdown(message["content"])
    else:
        st.chat_message("assistant").markdown(message["content"])

# Request user to input prompt
user_question = st.chat_input("Message Me: ")

# User submits the question
if user_question:
    # Show user message in the chat
    st.chat_message("user").markdown(user_question)

    # Get the chatbot's response
    result = chain_with_history.invoke(
        {"question": user_question},
        config={"configurable": {"session_id": st.session_state.session_id}},
    )
    
    # Show chatbot's response in the chat
    st.chat_message("assistant").markdown(result.content)
    
    # Save the conversation in chat history
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    st.session_state.chat_history.append({"role": "assistant", "content": result.content})

# When new conversation is initiated
if st.button("Start New Conversation"):
    # Delete chat history from the PostgreSQL database
    delete_chat_history(st.session_state.session_id)
    
    # Generate a new session ID
    st.session_state.session_id = str(uuid.uuid4())
    
    # Clear chat history in website's session state
    st.session_state.chat_history = []
    
    # Rerun the app
    st.rerun()
