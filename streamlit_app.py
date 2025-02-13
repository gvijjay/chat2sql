import io
import sys

import streamlit as st
import pandas as pd
import os
import plotly.graph_objs as go
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

USER = 'test_owner'
PASSWORD = 'tcWI7unQ6REA'
HOST = 'ep-yellow-recipe-a5fny139.us-east-2.aws.neon.tech:5432'
DATABASE = 'test'

def file_to_sql(file_path, table_name, user, password, host, db_name):
    import pandas as pd
    import os
    from sqlalchemy import create_engine

    # engine = create_engine(f"postgresql://{user}:{password}@{host}/{db_name}")
    engine = create_mysql_engine(user, password, host, db_name)

    if not table_name:
        table_name = os.path.splitext(os.path.basename(file_path))[0]

    file_extension = os.path.splitext(file_path)[-1].lower()
    if file_extension == '.xlsx':
        df = pd.read_excel(file_path)
    elif file_extension == '.csv':
        df = pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide an Excel (.xlsx) or CSV (.csv) file.")

    df.to_sql(table_name, con=engine, if_exists='replace', index=False)
    return df


def create_mysql_engine(user, password, host, db_name):
    from sqlalchemy import create_engine, text

    if db_name:
        connection_str = f'postgresql://{user}:{password}@{host}/{db_name}'
    else:
        connection_str = f'postgresql://{user}:{password}@{host}/'
    engine = create_engine(connection_str)
    return engine


# Function to generate code from OpenAI API
def generate_code(prompt_eng):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_eng}
        ]
    )
    all_text = ""
    for choice in response.choices:
        message = choice.message
        chunk_message = message.content if message else ''
        all_text += chunk_message
    print(all_text)
    if "```python" in all_text:
        code_start = all_text.find("```python") + 9
        code_end = all_text.find("```", code_start)
        code = all_text[code_start:code_end]
    else:
        code = all_text
    return code


def execute_py_code(code, df):
    # Local variables for executing the code
    local_vars = {'df': df}
    buffer = io.StringIO()
    sys.stdout = buffer

    try:
        # Execute the code
        exec(code, globals(), local_vars)

        # Check for a Plotly figure named 'fig'
        if 'fig' in local_vars and isinstance(local_vars['fig'], go.Figure):
            return local_vars['fig']

        # Otherwise, capture output or return last evaluated expression
        output = buffer.getvalue().strip()
        if output:
            return output
        else:
            last_line = code.strip().split('\n')[-1]
            return eval(last_line, globals(), local_vars)
    except Exception as e:
        return f"Error executing code: {e}"
    finally:
        sys.stdout = sys.__stdout__


def execute_py_code1(code, df):
    # Local variables for executing the code
    local_vars = {'df': df}
    buffer = io.StringIO()
    sys.stdout = buffer

    try:
        # Execute the code
        exec(code, globals(), local_vars)

        # If 'result' exists in local_vars, process it
        if 'result' in local_vars:
            result = local_vars['result']
            print("-------------------------------")
            print(result)
        else:
            # Otherwise, capture printed output or last evaluated expression
            output = buffer.getvalue().strip()
            if output:
                result = output
                print("result is,,,,,,,,,,,,,,,,", result)
            else:
                last_line = code.strip().split('\n')[-1]
                result = eval(last_line, globals(), local_vars)

        # Convert scalar, list, or dictionary into a DataFrame
        if isinstance(result, (int, float, str)):
            result = pd.DataFrame({'Value': [result]})
        elif isinstance(result, list):
            result = pd.DataFrame({'Values': result})
        elif isinstance(result, dict):
            result = pd.DataFrame([result])
        elif not isinstance(result, pd.DataFrame):
            raise ValueError("The result is not in a supported format (scalar, list, dict, or DataFrame).")

        # Ensure no thousands separators in numerical columns
        if isinstance(result, pd.DataFrame):
            for col in result.select_dtypes(include=['float', 'int']).columns:
                result[col] = result[col].apply(lambda x: x if pd.isnull(x) else float(x))

        print("result will be..", result)
        return result

    except Exception as e:
        return pd.DataFrame({'Error': [f"Error executing code: {e}"]})

    finally:
        sys.stdout = sys.__stdout__

def main():
    st.title("Talk2SQL")
    st.sidebar.header("Upload File")

    # Upload File
    uploaded_file = st.sidebar.file_uploader("Choose a file", type=["csv", "xlsx", "txt"])

    if uploaded_file is not None:
        # Save file temporarily
        upload_dir = "upload"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, uploaded_file.name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        table_name = uploaded_file.name.split('.')[0]
        df = file_to_sql(file_path, table_name, USER, PASSWORD, HOST, DATABASE)

        st.write("### Uploaded Data Preview")
        st.write(df.head())

        # Query Input
        user_query = st.text_input("Enter your query (e.g., 'Plot a scatter chart of X vs Y')")

        if user_query:
            # Define graph-related keywords
            graph_keywords = [
                "plot", "graph", "visualize", "visualization", "scatter", "bar chart",
                "line chart", "histogram", "pie chart", "bubble chart", "heatmap", "box plot",
                "generate chart", "create graph", "draw", "trend", "correlation"
            ]

            if any(keyword in user_query.lower() for keyword in graph_keywords):
                # Graph-related prompt
                metadata_str = ", ".join(df.columns.tolist())
                prompt_eng = (
                    f"You are an AI specialized in data analytics and visualization."
                    f"Data used for analysis is stored in a pandas DataFrame named `df`. "
                    f"The DataFrame `df` contains the following attributes: {metadata_str}. "
                    f"Based on the user's query, generate Python code using Plotly to create the requested type of graph. "
                    f"The graph must include a title, axis labels, and appropriate colors. "
                    f"Return a Plotly Figure object named 'fig'. The user's query is: {user_query}."
                )

                # Generate visualization code
                code = generate_code(prompt_eng)
                print(code)
                fig = execute_py_code(code, df)

                if isinstance(fig, go.Figure):
                    st.plotly_chart(fig)
                else:
                    st.error(f"Failed to generate a graph: {fig}")


            else:
                # Text-based prompt
                print("Else-Condition......")
                metadata_str = ", ".join(df.columns.tolist())
                prompt_eng = (
                    f"""
                You are a Python expert focused on answering user queries about data preprocessing and analysis. Strictly adhere to the following rules:
                
                1. Data-Driven Queries:
                    - Assume the `df` DataFrame in memory contains the following columns: {metadata_str}.
                    - Generate Python code that interacts directly with the `df` DataFrame to provide accurate results based strictly on the data.
                    - Ensure all outputs are converted into a pandas `DataFrame` and presented in **tabular format**.
                
                2. Output Format:
                    - Always assign the final output to a variable named `result` as a pandas `DataFrame`.
                    - Use meaningful column names to represent the result.
                    - Do not Format numerical outputs with commas as thousands separators. (e.g., `2,020` for 2020).
                    - Ensure text and non-numeric outputs remain unaltered.
                
                3. Scope:
                    - Do not make any assumptions or provide any example outputs.
                    - Exclude any visualization or plotting.
                
                4. Invalid or Non-Data Queries:
                    - If the query is unrelated to the dataset or cannot be answered, return an appropriate error message.
                
                Remember:
                - Use only the columns provided in the `df` DataFrame.
                - The output must always be in the form of a pandas `DataFrame` assigned to the variable `result`.
                User query: {user_query}.

            """
                )

                # Generate text-based response
                code = generate_code(prompt_eng)
                print(code)
                result = execute_py_code1(code, df)
                print("Final Result........",result)

                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    st.write("### Query Result")
                    display_without_commas(result)

def display_without_commas(result_df):
    # Format numbers as plain text without commas
    st.write(result_df.style.format(precision=0, formatter={col: '{:.0f}'.format for col in result_df.select_dtypes(include=['float','int']).columns}))


if __name__ == "__main__":
    main()
