from crewai.tools import BaseTool
import pandas as pd
import json
import yaml
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
import os
from typing import Optional, Type, Any
from pydantic import BaseModel, Field



class FixedEmbeddingToolSchema(BaseModel):
    """Input for EmbeddingTool"""

    pass


class EmbeddingToolSchema(FixedEmbeddingToolSchema):
    """Input for EmbeddingTool"""

    config_path: str = Field(..., description="Base config path containing paths to knowledge base and db")



class EmbeddingTool(BaseTool):
    name: str = "Create embedding for CSV and JSON files"
    description: str = (
        "A tool that can be used to embed the data from a CSV or JSON file and store the embedding into a database"
    )
    args_schema: Type[BaseModel] = EmbeddingToolSchema
    config_path: Optional[str] = None


    def __init__(self, config_path: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)

        if config_path is not None:
            self.config_path = config_path
            self.description = f"A tool that can be used to embed the knowledge in {config_path}'s content."
            self.args_schema = FixedEmbeddingToolSchema
            self._generate_description()    



    def _find_sqlite3_files(self, directory):
        # Iterate through all files and directories in the given path
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.sqlite3'):
                    print(f"Found .sqlite3 file: {os.path.join(root, file)}")
                    return True 
        return False


    def _run(self, **kwargs: Any) -> str:
        """This tool will be used in creating embeddings from the JSON and CSV data from the config file present in the input string and store those embeddings in a database"""

        base_config_path = kwargs.get("config_path", self.config_path)


        try:
            with open(base_config_path, 'r') as f:
                base_config = yaml.safe_load(f)
        except FileNotFoundError:
            return f"Error: Config file not found at {base_config_path}"
        except yaml.YAMLError as e:
            return f"Error: Could not parse config file at {base_config_path}: {e}"



        try:
            base_knowledge_config = base_config.get("knowledge_base")
            base_db_config = base_config.get("db")

            if not base_knowledge_config:
                return "Error: 'base_knowledge_config' must be provided."
            if not base_db_config:
                return "Error: 'base_db_config' must be provided."

            # Read config file
            csv_paths = base_knowledge_config.get("csv_paths", [])
            json_paths = base_knowledge_config.get("json_paths", [])


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


            db_path = base_db_config.get("chroma_db_path")

            if self._find_sqlite3_files(db_path):
                return f"ChromaDB already exists at {db_path}. Please delete the files to create a new database if you have new knowledge base."
            
            # Create embeddings and store in Chroma
            embeddings = OllamaEmbeddings(base_url="http://localhost:11434", model="mxbai-embed-large")

            
            # Ensure the directory exists
            os.makedirs(db_path, exist_ok=True)
            vectordb = Chroma.from_documents(documents=all_docs, embedding=embeddings, persist_directory=db_path)
            vectordb.persist()

            return f"Data successfully embedded and stored in Chroma database at '{db_path}'."

        except Exception as e:
            return f"An unexpected error occurred: {e}"
        
