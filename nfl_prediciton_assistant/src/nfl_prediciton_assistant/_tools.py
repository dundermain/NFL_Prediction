from crewai_tools import JSONSearchTool, CSVSearchTool

from crewai_tools import DirectoryReadTool


def json_to_embeddings(json_paths):

    json_search_tool = JSONSearchTool(

        json_path = json_paths,
        config={
            "llm": {
                "provider": "ollama",  # Other options include google, openai, anthropic, llama2, etc.
                "config": {
                    "model": "gemma2",
                    # Additional optional configurations can be specified here.
                    # temperature=0.5,
                    # top_p=1,
                    # stream=true,
                },
            },
            "embedder": {
                "provider": "ollama", # or openai, ollama, ...
                "config": {
                    "model": "mxbai-embed-large",
                    # Further customization options can be added here.
                },
            },
        }
    )

    return json_search_tool

def csv_to_embeddings(csv_paths):

    csv_search_tool = CSVSearchTool(

        csv_path = csv_paths,
        config={
            "llm": {
                "provider": "ollama",  # Other options include google, openai, anthropic, llama2, etc.
                "config": {
                    "model": "gemma2",
                    # Additional optional configurations can be specified here.
                    # temperature=0.5,
                    # top_p=1,
                    # stream=true,
                },
            },
            "embedder": {
                "provider": "ollama", # or openai, ollama, ...
                "config": {
                    "model": "mxbai-embed-large",
                    # Further customization options can be added here.
                },
            },
        }
    )

    return csv_search_tool

