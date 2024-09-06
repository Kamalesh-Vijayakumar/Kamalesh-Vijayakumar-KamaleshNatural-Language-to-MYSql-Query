import mysql.connector
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
import streamlit as st
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity

#App Title
st.title("Interactive MySQL Query Generator Chatbot")

base="dark"
primaryColor="#4bffd6"
backgroundColor="#0e1017"

# Database configuration
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'root',
    'database': 'kamalesh'
}

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embeddings(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

#Function to connect to the database and fetch data
@st.cache_data
def fetch_data(db_config, table_name):
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()

    query = f"SELECT * FROM {table_name}"
    cursor.execute(query)
    results = cursor.fetchall()
    columns = [i[0] for i in cursor.description]
    df = pd.DataFrame(results, columns=columns)

    cursor.close()
    conn.close()
    return df

# Fetch data from the database
df = fetch_data(db_config, 'titanic')

# Identify text columns
text_columns = df.select_dtypes(include=['object']).columns

# Convert text columns to vectors
@st.cache_data
def compute_text_embeddings(df, text_columns):
    text_embeddings = {}
    for column in text_columns:
        texts = df[column].astype(str).tolist()  # Convert column to list of strings
        embeddings = get_bert_embeddings(texts)
        text_embeddings[column] = embeddings
    return text_embeddings

text_embeddings = compute_text_embeddings(df, list(text_columns))

# Set up Gemini API

GEMINI_API_KEY = "use your gemini api key"  # Replace with your actual API key
genai.configure(api_key=GEMINI_API_KEY)


generation_config = {
    "temperature": 0,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}


# Define the prompt for column name replacement
system_instruction = (

    "You are provided with a schema of column names from the 'titanic' table. Your task is to replace terms related to columns in the input query "
    "with their exact column names from the schema. Make sure to follow these rules:\n\n"
    "Identify any terms in the query that correspond to the column names in the schema.\n"
    "Replace those terms with the exact column names from the schema, appending (column) to indicate the column name.\n"
    "Do not alter any other words or phrasing in the query.\n"
    "After generating the modified query, also extract and list the specific values or terms that are not column names from the query. "
    "Format these extracted terms as value(record).\n\n"
    "The table name you should always use is 'titanic'.\n\n"
    "Here is the schema:\n\n"
    "PassengerId\nSurvived\nPclass\nName\nSex\nAge\nSibSp\nParch\nTicket\nFare\nCabin\nEmbarked\n"
    "Input Query:\n\n{input_query}\n\n"
    "Output:\n\nModified Query:\nThe query with column names replaced and formatted correctly.\n"
    "Extracted Fields:\nA list of values or terms from the query that are not column names, formatted as value(record).\n"
)

gemini_model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction=system_instruction
)

# Function to vectorize a text
def vectorize_text(text):
    return get_bert_embeddings([text])

# Function to find best matches
def find_best_matches(column_embeddings, query_embedding):
    similarities = cosine_similarity(column_embeddings, query_embedding)
    best_match_idx = similarities.argmax()
    return best_match_idx

def adjust_query_with_best_matches(modified_query, best_matches):
    for column, best_match in best_matches.items():
        if best_match:
            # If the query uses a LIKE clause for names, adjust the pattern to use '%{best_match}%'
            if "LIKE" in modified_query and column.lower() == "name":
                modified_query = modified_query.replace(f"LIKE '{best_match}%'", f"LIKE '%{best_match}%'")
                modified_query = modified_query.replace(f"LIKE '{best_match}'", f"LIKE '%{best_match}%'")
            else:
                modified_query = modified_query.replace(f"{column}(column)", f"{best_match}(record)")
    return modified_query

# Example usage in the chatbot:
input_query = st.text_input("Ask a query:")

if input_query:
    try:
        # Start a new chat session
        chat_session = gemini_model.start_chat()
        response = chat_session.send_message(input_query)

        # Extract the modified query
        modified_user_query = response.text.strip('```').strip()

        # Vectorize the modified query
        modified_query_embedding = vectorize_text(modified_user_query)

        # Find the best matches for the modified query
        best_matches = {}
        for column in text_columns:
            best_match_idx = find_best_matches(text_embeddings[column], modified_query_embedding)
            best_matches[column] = df.iloc[best_match_idx][column]

        # Adjust the query with best matches
        adjusted_query = adjust_query_with_best_matches(modified_user_query, best_matches)

        # Display the adjusted query
        st.write("Adjusted Query:")
        st.write(adjusted_query)

        # Continue with MySQL query generation and execution...

        # Proceed with MySQL query generation and execution...


        # Define the prompt for MySQL conversion
        system_instruction_mysql = (
            "You are provided with a modified query in natural language that uses column names from a database schema. "
            "Your task is to convert this modified query into a valid MySQL query. "
            "When dealing with names or other text fields that might have multiple matches, use the LIKE operator with wildcard characters.\n\n"
            "Example Modified Query:\n"
            "Give me the passenger details whose name is Owen.\n\n"
            "MySQL Query:\n"
            "SELECT PassengerId, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked "
            "FROM kamalesh.titanic "
            "WHERE Name LIKE '%Owen%';\n\n"
            "if a colunm name comes with the natural language then always add LIKE '%name%' with the query \n"
            "Use this format to convert the following modified query into a MySQL query.\n\n"
            "{adjusted_query}"
        )

        gemini_model_mysql = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
            system_instruction=system_instruction_mysql
        )

        chat_session_mysql = gemini_model_mysql.start_chat()
        mysql_query_response = chat_session_mysql.send_message(adjusted_query)

        # Extract and clean the MySQL query
        mysql_query = mysql_query_response.text.strip('```').strip()
        if mysql_query.startswith("sql"):
            mysql_query = mysql_query[3:].strip()

        st.write("Generated MySQL Query:")
        st.code(mysql_query, language="sql")

        # Execute the generated MySQL query
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        cursor.execute(mysql_query)
        query_results = cursor.fetchall()
        query_columns = [i[0] for i in cursor.description]

        # Convert query results to a DataFrame
        query_df = pd.DataFrame(query_results, columns=query_columns)

        st.write("Query Results:")
        st.write(query_df)

    except mysql.connector.Error as err:
        st.error(f"Error executing query: {err}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
