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
    user_query: str = Field(..., description="User query")



class RetrievalTool(BaseTool):
    name: str = "Retrieve the relevant documents and information" 
    description: str = (
        "A tool that can be used to retrieve the relevant documents and information from database"
    )
    args_schema: Type[BaseModel] = RetrievalToolSchema
    config_path: Optional[str] = None
    user_query: Optional[str] = None


    def __init__(self, config_path: Optional[str] = None, user_query: Optional[str] = None,**kwargs):
        super().__init__(**kwargs)

        if config_path is not None:
            self.config_path = config_path
            self.user_query = user_query
            self.description = f"A tool that can be used to retrieve the relevant documents and information from database in {config_path}'s db path."
            self.args_schema = FixedRetrievalToolSchema
            self._generate_description()    


    def _run(self, **kwargs: Any) -> str:
        """This tool will be used in retrieving embeddings from the database"""

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

            user_query = kwargs.get("user_query", self.user_query)
            # user_query = "injuries"

            relevant_info = vector_db.similarity_search_by_vector(embedding=embeddings.embed_query(user_query))

            return relevant_info


        except Exception as e:
            return f"An unexpected error occurred: {e}"
        