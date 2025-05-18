import os
import logging
from typing import List, Tuple
from utils.load_config import LoadConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

APPCFG = LoadConfig()


class ChatBot:
    """
    A unified ChatBot class that uses RAG with document embeddings for Q&A.
    Supports uploaded CSV/XLSX files converted to embeddings in ChromaDB.
    """
    
    @staticmethod
    def respond(chatbot: List, message: str, chat_type: str, app_functionality: str) -> Tuple:
        """
        Respond to a message using unified RAG approach with intelligent retrieval.

        Args:
            chatbot (List): A list representing the chatbot's conversation history.
            message (str): The user's input message to the chatbot.
            chat_type (str): Chat type (unified to single RAG approach).
            app_functionality (str): Application functionality mode.

        Returns:
            Tuple[str, List]: Empty string and updated chatbot conversation list.
        """
        if app_functionality == "Chat":
            logger.info(f"Processing chat message: {message[:50]}...")
            
            # Special commands for system information
            if message.lower().strip() in ['status', 'info', '/status', '/info']:
                collection_info = ChatBot._get_collection_status()
                chatbot.append((message, collection_info))
                return "", chatbot
            
            try:
                # Check if ChromaDB collection exists
                if not ChatBot._check_collection_exists():
                    error_msg = """📭 **No Documents Found**
                    
Please upload your CSV/XLSX files first:
1. Switch to '🔧 Process files' mode  
2. Click '📁 Upload CSV/XLSX Files' button
3. Select your files
4. Wait for processing to complete
5. Switch back to '🔧 Chat' mode
6. Ask your questions!

**Tip**: Type 'status' to check current system status."""
                    logger.warning("No ChromaDB collection found")
                    chatbot.append((message, error_msg))
                    return "", chatbot
                
                # Rest of the existing code...
                
                # Step 1: Use LLM to determine what should be retrieved
                logger.info("Step 1: Determining retrieval strategy with LLM")
                retrieval_query = ChatBot._generate_retrieval_query(message)
                logger.info(f"Generated retrieval query: {retrieval_query}")
                
                # Step 2: Get embeddings for the retrieval query
                logger.info("Step 2: Generating embeddings for retrieval")
                query_embeddings = APPCFG.embeddings_client.embed_query(retrieval_query)
                logger.info("Embeddings generated successfully")
                
                # Step 3: Retrieve relevant documents from ChromaDB
                logger.info("Step 3: Retrieving relevant documents from ChromaDB")
                vectordb = APPCFG.chroma_client.get_collection(name=APPCFG.collection_name)
                results = vectordb.query(
                    query_embeddings=[query_embeddings],
                    n_results=APPCFG.top_k
                )
                logger.info(f"Retrieved {len(results['documents'][0])} relevant documents")
                
                # Step 4: Format retrieved documents
                formatted_docs = ChatBot._format_retrieved_documents(results)
                logger.info("Documents formatted for LLM processing")
                
                # Step 5: Generate final response using LLM with retrieved context
                logger.info("Step 5: Generating final response with context")
                response = ChatBot._generate_final_response(message, formatted_docs)
                logger.info("Response generated successfully")
                
                # Log statistics
                ChatBot._log_statistics(results, response)
                
            except Exception as e:
                logger.error(f"Error processing message: {str(e)}")
                response = f"I encountered an error processing your request: {str(e)}. Please try again or upload new documents."
            
            chatbot.append((message, response))
            return "", chatbot
        
        else:
            logger.info(f"Non-chat functionality: {app_functionality}")
            return "", chatbot
    
    @staticmethod
    def _check_collection_exists() -> bool:
        """Check if the ChromaDB collection exists and has documents."""
        try:
            collection = APPCFG.chroma_client.get_collection(name=APPCFG.collection_name)
            count = collection.count()
            logger.info(f"Collection '{APPCFG.collection_name}' found with {count} documents")
            return count > 0
        except Exception as e:
            logger.warning(f"Collection check failed: {str(e)}")
            return False
    
    @staticmethod
    def _generate_retrieval_query(user_message: str) -> str:
        """Use LLM to generate an optimized query for document retrieval."""
        retrieval_prompt = f"""
        Based on the user's question, generate an optimized search query that will help retrieve the most relevant documents from a database of CSV/XLSX files.
        
        User's question: {user_message}
        
        Generate a focused search query that captures the key concepts and entities the user is asking about. Make it specific enough to find relevant data but broad enough to not miss important information.
        
        Search query:"""
        
        messages = [
            {"role": "system", "content": "You are an expert at generating search queries for document retrieval. Generate concise, focused queries."},
            {"role": "user", "content": retrieval_prompt}
        ]
        
        try:
            llm_response = APPCFG.azure_openai_client.invoke(messages)
            retrieval_query = llm_response.content.strip()
            
            # Fallback to user message if LLM response is too short or empty
            if len(retrieval_query) < 3:
                retrieval_query = user_message
                
            return retrieval_query
        except Exception as e:
            logger.warning(f"Failed to generate retrieval query, using original message: {str(e)}")
            return user_message
    
    @staticmethod
    def _format_retrieved_documents(results) -> str:
        """Format the retrieved documents for better LLM processing."""
        if not results['documents'] or not results['documents'][0]:
            return "No relevant documents found."
        
        formatted_docs = "Retrieved Documents:\n\n"
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0], 
            results['metadatas'][0], 
            results['distances'][0]
        )):
            formatted_docs += f"--- Document {i+1} (Source: {metadata.get('source', 'Unknown')}, Relevance: {1-distance:.3f}) ---\n"
            formatted_docs += f"{doc}\n\n"
        
        return formatted_docs
    
    @staticmethod
    def _generate_final_response(user_message: str, context_docs: str) -> str:
        """Generate the final response using the user's question and retrieved context."""
        
        # Check if the retrieved documents seem relevant to the question
        relevance_check_prompt = f"""
        Quickly assess if the following retrieved documents are relevant to the user's question.
        
        User's question: {user_message}
        Retrieved documents: {context_docs[:500]}...
        
        Are these documents relevant to answer the user's question? Answer with just "RELEVANT" or "NOT_RELEVANT" and a brief explanation.
        """
        
        try:
            relevance_response = APPCFG.azure_openai_client.invoke([
                {"role": "system", "content": "You are a relevance checker. Assess if retrieved documents match the user's question."},
                {"role": "user", "content": relevance_check_prompt}
            ])
            
            if "NOT_RELEVANT" in relevance_response.content:
                logger.warning("Retrieved documents not relevant to query")
                return f"""❌ **No Relevant Information Found**

The documents in the system don't contain information about your question: "{user_message}"

**Current Data Contains**: {ChatBot._get_data_summary()}

**Suggestions:**
1. Upload files that contain the information you're looking for
2. Rephrase your question to match the available data
3. Type 'status' to see what data is currently available

**Available Commands:**
- `status` - Check what data is loaded
- Upload new files in 'Process files' mode"""
        except Exception as e:
            logger.warning(f"Relevance check failed: {str(e)}")
        
        # Generate final response if documents are relevant
        final_prompt = f"""
        Based on the user's question and the relevant documents retrieved from the database, provide a comprehensive and accurate answer.
        
        User's question: {user_message}
        
        {context_docs}
        
        Instructions:
        - Use only the information provided in the retrieved documents
        - If the documents don't contain enough information to answer the question, state this clearly
        - Be specific and cite relevant details from the documents
        - If you find conflicting information, mention this
        - Provide a clear, well-structured answer
        - Format your response with markdown for better readability
        
        Answer:"""
        
        messages = [
            {"role": "system", "content": str(APPCFG.rag_llm_system_role)},
            {"role": "user", "content": final_prompt}
        ]
        
        try:
            llm_response = APPCFG.azure_openai_client.invoke(messages)
            return llm_response.content
        except Exception as e:
            logger.error(f"Failed to generate final response: {str(e)}")
            return f"I apologize, but I encountered an error generating a response: {str(e)}"
    
    @staticmethod
    def _get_data_summary() -> str:
        """Get a brief summary of what data is currently available."""
        try:
            collection = APPCFG.chroma_client.get_collection(name=APPCFG.collection_name)
            sample = collection.query(
                query_embeddings=[[0.0] * 1536],
                n_results=1
            )
            
            if sample['metadatas'] and sample['metadatas'][0]:
                source = sample['metadatas'][0][0].get('source', 'Unknown dataset')
                return f"Data from '{source}' with {collection.count()} records"
            return "Unknown dataset"
        except Exception:
            return "No data available"
    
    @staticmethod
    def _get_collection_status() -> str:
        """Get detailed status of the current collection."""
        try:
            collection = APPCFG.chroma_client.get_collection(name=APPCFG.collection_name)
            count = collection.count()
            
            # Get sample documents to understand the data
            sample = collection.query(
                query_embeddings=[[0.0] * 1536],  # Zero vector to get any documents
                n_results=min(3, count)
            )
            
            # Analyze the sample documents
            sources = set()
            sample_content = []
            
            if sample['metadatas'] and sample['documents']:
                for metadata, doc in zip(sample['metadatas'][0], sample['documents'][0]):
                    sources.add(metadata.get('source', 'Unknown'))
                    sample_content.append(doc[:100] + "..." if len(doc) > 100 else doc)
            
            status_msg = f"""📊 **System Status**

**Collection Info:**
- Name: `{APPCFG.collection_name}`
- Total Documents: **{count}**
- Data Sources: **{', '.join(sources)}**

**Sample Data Preview:**
```
{sample_content[0] if sample_content else 'No data available'}
```

**Ready for Questions!** 
You can now ask questions about:
- Data analysis and statistics
- Specific information from your files
- Trends and patterns in the data

**Examples:**
- "What are the main columns in my data?"
- "Show me some statistics"
- "What trends can you find?"
"""
            return status_msg
            
        except Exception as e:
                        logger.warning(f"Failed to log statistics: {str(e)}")
            
    @staticmethod
    def _log_statistics(results, response):
        """Log statistics about the retrieval and response."""
        try:
            num_docs_retrieved = len(results['documents'][0]) if results['documents'] else 0
            avg_distance = sum(results['distances'][0]) / len(results['distances'][0]) if results['distances'] and results['distances'][0] else 0
            response_length = len(response.split())
            
            logger.info(f"Statistics - Documents retrieved: {num_docs_retrieved}, "
                       f"Average relevance: {1-avg_distance:.3f}, "
                       f"Response length: {response_length} words")
        except Exception as e:
            logger.warning(f"Failed to log statistics: {str(e)}")