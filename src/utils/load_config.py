import os
import logging
from dotenv import load_dotenv
import yaml
from pyprojroot import here
import shutil
from langchain_openai import ChatOpenAI
import chromadb
from langchain_openai import OpenAIEmbeddings

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("Environment variables are loaded:", load_dotenv())


class LoadConfig:
    def __init__(self) -> None:
        logger.info("Initializing LoadConfig")
        
        with open(here("configs/app_config.yml")) as cfg:
            app_config = yaml.load(cfg, Loader=yaml.FullLoader)

        self.load_directories(app_config=app_config)
        self.load_llm_configs(app_config=app_config)
        self.load_openai_models()
        self.load_chroma_client()
        self.load_rag_config(app_config=app_config)

        # Clean up any existing collections for fresh start
        # Uncomment the line below if you want to clean up on every restart
        # self.cleanup_existing_collections()
        
        logger.info("LoadConfig initialization completed")

    def load_directories(self, app_config):
        """Load all directory configurations."""
        logger.info("Loading directory configurations")
        
        self.stored_csv_xlsx_directory = here(
            app_config["directories"]["stored_csv_xlsx_directory"])
        self.persist_directory = app_config["directories"]["persist_directory"]
        
        # Create persist directory if it doesn't exist
        persist_path = str(here(self.persist_directory))
        if not os.path.exists(persist_path):
            os.makedirs(persist_path)
            logger.info(f"Created persist directory: {persist_path}")

    def load_llm_configs(self, app_config):
        """Load LLM configurations."""
        logger.info("Loading LLM configurations")
        
        self.model_name = os.getenv("gpt_deployment_name", "gpt-4o-mini")
        self.temperature = app_config["llm_config"]["temperature"]
        self.base_url = app_config["llm_config"]["base_url"]
        self.embedding_model_name = "text-embedding-3-small"
        
        # Enhanced system roles for unified RAG system
        self.rag_llm_system_role = """
        You are a friendly and knowledgeable AI assistant who helps users analyze and understand their uploaded CSV/XLSX files. You’re not just a data expert—you’re also a helpful and engaging chatbot! Your responsibilities include:

            Warmly greeting users and responding conversationally—even to casual messages like “hi” or “how are you?”

            Analyzing and interpreting data from uploaded spreadsheets to provide clear and accurate insights.

            Giving helpful, easy-to-understand answers based on the data provided—always prioritizing clarity.

            Being transparent when information isn’t available in the uploaded documents. Say so clearly and kindly.

            Offering structured responses with relevant charts, summaries, and key findings when needed.

            Suggesting useful follow-up questions, further analysis, or ways to get more value from the data.

            Always acting like a chatbot—conversational, responsive, and never stiff or robotic.

            Never say “I can only chat about your data”—you’re here to assist and chat like a helpful companion.

            You are friendly, professional, curious, and supportive. Your goal is to make working with data easy and enjoyable.
            """
        logger.info("LLM configurations loaded successfully")

    def load_openai_models(self):
        """Load and configure OpenAI models."""
        logger.info("Loading OpenAI models")
        
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            logger.error("OPENAI_API_KEY not found in environment variables")
            raise ValueError("OPENAI_API_KEY is required")

        try:
            # Main LLM for chat responses
            self.azure_openai_client = ChatOpenAI(
                model="gpt-4o-mini",
                api_key=openai_api_key,
                temperature=self.temperature,
                base_url=self.base_url,
            )
            logger.info("Main LLM client initialized")

            # Embeddings client for document processing
            self.embeddings_client = OpenAIEmbeddings(
                model=self.embedding_model_name,
                api_key=openai_api_key,
                base_url=self.base_url,
            )
            logger.info("Embeddings client initialized")

            # LangChain LLM for auxiliary tasks
            self.langchain_llm = ChatOpenAI(
                model="gpt-4o-mini",
                api_key=openai_api_key,
                temperature=self.temperature,
                base_url=self.base_url,
            )
            logger.info("LangChain LLM client initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI models: {str(e)}")
            raise

    def load_chroma_client(self):
        """Load and configure ChromaDB client."""
        logger.info("Loading ChromaDB client")
        
        try:
            persist_path = str(here(self.persist_directory))
            self.chroma_client = chromadb.PersistentClient(path=persist_path)
            
            # Log existing collections
            collections = self.chroma_client.list_collections()
            if collections:
                logger.info(f"Found {len(collections)} existing collections: {[c.name for c in collections]}")
            else:
                logger.info("No existing collections found")
                
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {str(e)}")
            raise

    def load_rag_config(self, app_config):
        """Load RAG-specific configurations."""
        logger.info("Loading RAG configurations")
        
        self.collection_name = app_config["rag_config"]["collection_name"]
        self.top_k = app_config["rag_config"]["top_k"]
        
        # Add batch processing configurations
        self.embedding_batch_size = app_config["rag_config"].get("embedding_batch_size", 100)
        self.storage_batch_size = app_config["rag_config"].get("storage_batch_size", 500)
        
        # Ensure top_k is reasonable
        if self.top_k > 20:
            logger.warning(f"top_k value {self.top_k} is quite high, consider reducing for better performance")
        
        # Validate batch sizes
        if self.embedding_batch_size > 1000:
            logger.warning(f"embedding_batch_size {self.embedding_batch_size} is very large, may cause rate limiting")
            self.embedding_batch_size = 100
            
        if self.storage_batch_size > 1000:
            logger.warning(f"storage_batch_size {self.storage_batch_size} is very large, reducing to 1000")
            self.storage_batch_size = 1000
        
        logger.info(f"RAG config loaded - Collection: {self.collection_name}, Top-K: {self.top_k}")
        logger.info(f"Batch sizes - Embedding: {self.embedding_batch_size}, Storage: {self.storage_batch_size}")

    def cleanup_existing_collections(self):
        """Clean up existing ChromaDB collections for a fresh start."""
        try:
            collections = self.chroma_client.list_collections()
            for collection in collections:
                if collection.name == self.collection_name:
                    self.chroma_client.delete_collection(name=collection.name)
                    logger.info(f"Deleted existing collection: {collection.name}")
        except Exception as e:
            logger.warning(f"Failed to cleanup collections: {str(e)}")

    def get_collection_info(self):
        """Get information about the current collection."""
        try:
            collection = self.chroma_client.get_collection(name=self.collection_name)
            count = collection.count()
            logger.info(f"Collection '{self.collection_name}' contains {count} documents")
            return {"exists": True, "count": count}
        except Exception:
            logger.info(f"Collection '{self.collection_name}' does not exist")
            return {"exists": False, "count": 0}

    def remove_directory(self, directory_path: str):
        """
        Remove the specified directory and all its contents.

        Parameters:
            directory_path (str): The path of the directory to be removed.

        Returns:
            None
        """
        if os.path.exists(directory_path):
            try:
                shutil.rmtree(directory_path)
                logger.info(f"Successfully removed directory: {directory_path}")
            except OSError as e:
                logger.error(f"Error removing directory {directory_path}: {e}")
        else:
            logger.info(f"Directory does not exist: {directory_path}")