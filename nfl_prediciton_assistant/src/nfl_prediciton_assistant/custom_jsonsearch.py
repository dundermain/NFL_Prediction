from crewai_tools import JSONSearchTool
import yaml


def JSONSearch(config_path):

    with open(config_path, 'r') as f:
        base_json_config = yaml.safe_load(f)

    json_knowledge_list = base_json_config.get("knowledge_base").get("csv_paths", [])

    for json_path in json_knowledge_list:

        json_searchtool = JSONSearchTool(

            website = json_path,
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

    return json_searchtool



