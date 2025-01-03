import os
import requests
from bs4 import BeautifulSoup
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter

# Directory to store the scraped text
SCRAPED_TEXT_DIR = "scraped_text"
os.makedirs(SCRAPED_TEXT_DIR, exist_ok=True)

# List of websites to scrape
WEBSITES = [
    "https://www.nfl.com/scores",
    "https://www.espn.com/nfl/scoreboard",
    "https://www.cbssports.com/nfl/scores/"
]

def scrape_website(url):
    """Scrapes the given website for NFL game information and links."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extracting text content (simplified for demonstration)
        text = soup.get_text(separator="\n")

        # Extracting links
        links = [a['href'] for a in soup.find_all('a', href=True)]
        links_text = "\n".join(links)

        # Combine text and links
        combined_text = text + "\n\nLinks:\n" + links_text
        return combined_text
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

def save_text_to_file(filename, text):
    """Saves the given text to a file."""
    filepath = os.path.join(SCRAPED_TEXT_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(text)
    return filepath

def process_and_store_embeddings():
    """Processes scraped text, creates embeddings, and stores them in ChromaDB."""
    # Initialize embeddings model
    embeddings_model = OllamaEmbeddings(base_url="http://localhost:11434", model="mxbai-embed-large")

    # Initialize ChromaDB with the embedding function
    vectorstore = Chroma(persist_directory="db", embedding_function=embeddings_model)

    # Process each text file in the scraped_text directory
    for filename in os.listdir(SCRAPED_TEXT_DIR):
        filepath = os.path.join(SCRAPED_TEXT_DIR, filename)

        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()

        # Split text into chunks for embeddings
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_text(text)

        # Create embeddings for each chunk explicitly
        embeddings = [embeddings_model._embed(chunk) for chunk in chunks]

        # Add texts and their embeddings to the vector store
        metadata = {"source": filename}
        vectorstore.add_texts(texts=chunks, embeddings=embeddings, metadatas=[metadata] * len(chunks))

    # Persist the vector store
    vectorstore.persist()

if __name__ == "__main__":
    for website in WEBSITES:
        print(f"Scraping {website}...")
        scraped_text = scrape_website(website)
        if scraped_text:
            # Save the scraped text to a file
            filename = website.replace("https://", "").replace("/", "_") + ".txt"
            save_text_to_file(filename, scraped_text)

    print("Processing and storing embeddings...")
    process_and_store_embeddings()
    print("Embeddings stored successfully.")
