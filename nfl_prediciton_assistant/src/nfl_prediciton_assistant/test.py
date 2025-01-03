from crewai.tools import BaseTool
from typing import Dict
import pandas as pd
import json
import yaml
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
import os

from typing import Optional


from crewai_tools import JSONSearchTool





def _run(input_data: Dict[str, str]) -> str:
    """This tool will be used in creating embeddings from the JSON and CSV data from the config file present in the input string and store those embeddings in a database"""

    try:
        knowledge_config_path = input_data.get("knowledge_config_path")
        db_config_path = input_data.get("db_config_path")

        if not knowledge_config_path:
            return "Error: 'knowledge_config_path' must be provided."
        if not db_config_path:
            return "Error: 'db_config_path' must be provided."

        # Read config file
        try:
            with open(knowledge_config_path, 'r') as f:
                knowledge_config = yaml.safe_load(f)
        except FileNotFoundError:
            return f"Error: Config file not found at {knowledge_config_path}"
        except yaml.YAMLError as e:
            return f"Error: Could not parse config file at {knowledge_config_path}: {e}"

        csv_paths = knowledge_config.get("csv_paths", [])
        json_paths = knowledge_config.get("json_paths", [])

        print(csv_paths, json_paths)

        if not csv_paths or not json_paths:
            return "Error: Both 'csv' and 'json' paths must be defined in the config file."

        # Read CSV
        for csv_path in csv_paths:
            try:
                df_csv = pd.read_csv(csv_path)
                csv_docs = [Document(page_content=str(row.to_dict()), metadata={"source": "csv", "row_index": i}) for i, row in df_csv.iterrows()]
            except FileNotFoundError:
                return f"Error: CSV file not found at {csv_path}"
            except pd.errors.ParserError:
                return f"Error: Could not parse CSV file at {csv_path}. Check the file format."

        # Read JSON
        for json_path in json_paths:
            try:
                with open(json_path, 'r') as f:
                    json_data = json.load(f)
                if isinstance(json_data, list):
                    json_docs = [Document(page_content=json.dumps(item), metadata={"source": "json", "item_index": i}) for i, item in enumerate(json_data)]
                elif isinstance(json_data, dict):
                    json_docs = [Document(page_content=json.dumps(json_data), metadata={"source": "json"})]
                else:
                    return f"Error: JSON file at {json_path} contains unexpected data structure. It should be a list or a dict."
                
            except FileNotFoundError:
                return f"Error: JSON file not found at {json_path}"
            except json.JSONDecodeError:
                return f"Error: Could not parse JSON file at {json_path}. Check the file format."
            

        all_docs = csv_docs + json_docs


        try:
            with open(db_config_path, 'r') as f:
                db_config = yaml.safe_load(f)
        except FileNotFoundError:
            return f"Error: Config file not found at {knowledge_config_path}"
        except yaml.YAMLError as e:
            return f"Error: Could not parse config file at {knowledge_config_path}: {e}"

        db_path = db_config.get("chroma_db_path")
        # Create embeddings and store in Chroma
        embeddings = OllamaEmbeddings(base_url="http://localhost:11434", model="mxbai-embed-large")
        # Ensure the directory exists
        os.makedirs(db_path, exist_ok=True)
        vectordb = Chroma.from_documents(documents=all_docs, embedding=embeddings, persist_directory=db_path)
        vectordb.persist()

        return f"Data successfully embedded and stored in Chroma database at {db_path}."

    except Exception as e:
        return f"An unexpected error occurred: {e}"
    

def retreive_documents(user_query, db_path):

    embeddings = OllamaEmbeddings(base_url="http://localhost:11434", model="mxbai-embed-large")

    vector_db = Chroma(persist_directory = db_path, embedding_function= embeddings)

    relevant_info = vector_db.similarity_search(user_query)

    return relevant_info
    

if __name__ == "__main__":
    tool = _run(input_data = {"knowledge_config_path": "src/nfl_prediciton_assistant/config/knowledge.yaml", "db_config_path": "src/nfl_prediciton_assistant/config/db_config.yaml"})
    print(retreive_documents(user_query='update on jaguars', db_path='db'))