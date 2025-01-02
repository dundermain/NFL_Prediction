from crewai_tools import WebsiteSearchTool
import yaml


def WebSearch(config_path):

    with open(config_path, 'r') as f:
        web_addresses = yaml.safe_load(f)

    web_address_dict = web_addresses['nfl_websites']

    for address in web_address_dict:
        web_address = web_address_dict[address]


        tool = WebsiteSearchTool(

            website = web_address,
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

    return tool



