import os
from crewai import Agent, Task, Crew
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"  # Replace with your actual key

# Initialize embeddings and vector store
embeddings = OpenAIEmbeddings()
db = None  # Initialize as None

def embed_data(data):
    """Embeds data and stores it in a vector database."""
    global db  # Access the global db variable
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.create_documents([data])
    db = FAISS.from_documents(docs, embeddings)
    return "Data embedded and stored."

def retrieve_data(query):
    """Retrieves relevant data from the vector database based on a query."""
    if db is None:
        return "No data has been embedded yet. Please embed data first."
    docs = db.similarity_search(query)
    return "\n".join([doc.page_content for doc in docs])

def generate_report(documents):
    """Generates a detailed report based on retrieved documents."""
    if not documents:
      return "No documents provided to generate report."
    return f"Report based on retrieved documents:\n{documents}"

# Define the agents
embedding_agent = Agent(
    role="Data Embedder",
    goal="Embed given data into a vector database for efficient retrieval.",
    llm="gpt-3.5-turbo"
)

retrieval_agent = Agent(
    role="Data Retriever",
    goal="Retrieve relevant information from the vector database based on user queries.",
    llm="gpt-3.5-turbo"
)

reporting_agent = Agent(
    role="Report Generator",
    goal="Create detailed reports based on the retrieved information.",
    llm="gpt-3.5-turbo"
)

# Sample data (replace with your actual data)
sample_data = """
Artificial intelligence (AI) is rapidly transforming various industries. 
Recent advancements in deep learning have led to breakthroughs in natural language processing and computer vision.
Large language models (LLMs) are becoming increasingly powerful, enabling more sophisticated applications like chatbots and automated content generation.
AI also plays a crucial role in healthcare, finance, and transportation, leading to more efficient and personalized services.
"""

# Define the tasks
embedding_task = Task(
    description="Embed the provided data into the vector database.",
    agent=embedding_agent,
    function=embed_data,
    function_kwargs={"data": sample_data}
)

retrieval_task = Task(
    description="Retrieve relevant information based on the user's query.",
    agent=retrieval_agent,
    function=retrieve_data,
    context="User Query: What are the recent advancements in AI?", # Example user query
    # human_input=True # Uncomment to allow user input for the query at runtime
)

reporting_task = Task(
    description="Generate a detailed report based on the retrieved information.",
    agent=reporting_agent,
    function=generate_report,
    context=retrieval_task.output # Pass output of retrieval task as input to reporting task
)

# Create the crew and execute the tasks
crew = Crew(
    agents=[embedding_agent, retrieval_agent, reporting_agent],
    tasks=[embedding_task, retrieval_task, reporting_task],
    verbose=True
)

result = crew.kickoff()
print("######################")
print(result)