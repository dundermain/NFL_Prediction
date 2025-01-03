from crewai_tools import CSVSearchTool
import yaml


def CSVSearch(config_path):

    with open(config_path, 'r') as f:
        base_csv_config = yaml.safe_load(f)

    csv_knowledge_list = base_csv_config.get("knowledge_base").get("csv_paths", [])

    for csv_path in csv_knowledge_list:

        csv_searchtool = CSVSearchTool(

            website = csv_path,
            summarize = True,

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

    return csv_searchtool



