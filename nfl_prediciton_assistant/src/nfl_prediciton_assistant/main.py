#!/usr/bin/env python
import sys
import warnings

from nfl_prediciton_assistant.crew import NflPredicitonAssistant

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the crew.
    """
    print("Welcome to the NFL Agent! Type 'exit' to quit.")

    i = 0

    while i<1:
        # user_query = input("\nHow can I help you today?: ")
        user_query = "jaguars vs panthers"

        user_input = {'Question': user_query}

        i += 1



        if user_query.lower() == "exit":
            print("Goodbye! ")
            break
        try:
            response = NflPredicitonAssistant().crew().kickoff(inputs=user_input)
            print(f"\nResponse:\n{response}")

        except Exception as e:
            print(f"Error: {e}")

    


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "topic": "AI LLMs"
    }
    try:
        NflPredicitonAssistant().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        NflPredicitonAssistant().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "topic": "AI LLMs"
    }
    try:
        NflPredicitonAssistant().crew().test(n_iterations=int(sys.argv[1]), openai_model_name=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")
