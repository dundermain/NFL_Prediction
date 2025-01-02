from crewai.tools import BaseTool
import yaml
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from typing import Optional, Type, Any
from pydantic import BaseModel, Field



class FixedRetrievalToolSchema(BaseModel):
    """Input for RetrievalTool"""

    pass


class RetrievalToolSchema(FixedRetrievalToolSchema):
    """Input for RetrievalTool"""

    config_path: str = Field(..., description="Base config path containing paths to knowledge base and db")



class RetrievalTool(BaseTool):
    name: str = "Create embedding for CSV and JSON files"
    description: str = (
        "A tool that can be used to embed the data from a CSV or JSON file and store the embedding into a database"
    )
    args_schema: Type[BaseModel] = RetrievalToolSchema
    config_path: Optional[str] = None


    def __init__(self, config_path: Optional[str] = None, user_query: Optional[str] = None,**kwargs):
        super().__init__(**kwargs)

        if config_path is not None:
            self.config_path = config_path
            self.description = f"A tool that can be used to embed the knowledge in {config_path}'s content."
            self.args_schema = FixedRetrievalToolSchema
            self._generate_description()    


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

            base_db_config = base_config.get("db")

            if not base_db_config:
                return "Error: 'base_db_config' must be provided."

            db_path = base_db_config.get("chroma_db_path")
 
            embeddings = OllamaEmbeddings(base_url="http://localhost:11434", model="mxbai-embed-large")

            vector_db = Chroma(persist_directory = db_path, embedding_function= embeddings)

            relevant_info = vector_db.similarity_search("Jaguars")

            return relevant_info


        except Exception as e:
            return f"An unexpected error occurred: {e}"
        