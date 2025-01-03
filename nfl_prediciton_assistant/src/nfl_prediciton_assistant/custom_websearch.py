from crewai_tools import WebsiteSearchTool
import yaml
import time


def WebSearch(config_path):

    with open(config_path, 'r') as f:
        web_addresses = yaml.safe_load(f)

    web_address_dict = web_addresses['nfl_websites']

    for address in web_address_dict:
        web_address = web_address_dict[address]

        print(web_address)

        try:
            tool = WebsiteSearchTool(

                website = web_address,
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
        except Exception as e:
            print(f"Error: {e}")
            continue


        time.sleep(5)

    return tool



