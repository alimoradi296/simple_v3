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
    An enhanced ChatBot class that handles both document-specific queries and general conversations.
    Uses RAG when documents are available and falls back to general conversation mode otherwise.
    """
    
    @staticmethod
    def get_welcome_message() -> str:
        """Get a welcome message for new users."""
        has_docs = ChatBot._check_collection_exists()
        
        if has_docs:
            doc_summary = ChatBot._get_brief_collection_summary()
            return f"""👋 **Welcome back!** 

I'm your AI assistant, ready to help with anything you need! 

**🎯 What I can do:**
📊 **Document Analysis** - I have access to your uploaded documents ({doc_summary}) and can answer questions about them
💬 **General Chat** - Ask me about any topic you're curious about
🔍 **Information & Learning** - Get explanations, advice, or creative help

**💡 Try asking:**
- "What's in my uploaded data?"
- "Explain quantum physics to me"
- "Help me brainstorm ideas for..."
- "What are the trends in my dataset?"

What would you like to talk about today? 😊"""
        else:
            return f"""👋 **Hello there!** 

I'm your AI assistant, and I'm excited to chat with you! 

**🎯 I can help you with:**
💬 **Conversations** - Chat about any topic you're interested in
📊 **Document Analysis** - Upload CSV/XLSX files and I'll analyze them for you
🧠 **Learning & Explanations** - Ask me to explain concepts or help with problems
✨ **Creative Tasks** - Brainstorming, writing, or problem-solving

**💡 Getting started:**
- Just start chatting about anything on your mind!
- Use the sidebar (📁 button) to upload data files
- Ask me questions, seek advice, or explore topics

What would you like to talk about today? 😊"""

    @staticmethod
    def respond(chatbot: List, message: str, chat_type: str, app_functionality: str) -> Tuple:
        """
        Enhanced respond method that handles both document queries and general conversation.

        Args:
            chatbot (List): A list representing the chatbot's conversation history.
            message (str): The user's input message to the chatbot.
            chat_type (str): Type of chat interaction.
            app_functionality (str): Application functionality mode.

        Returns:
            Tuple[str, List]: Empty string and updated chatbot conversation list.
        """
        if app_functionality == "Chat":
            logger.info(f"Processing message: {message[:50]}...")
            
            # Handle first-time users with welcome message
            if not chatbot and not message.strip():
                welcome_msg = ChatBot.get_welcome_message()
                chatbot.append(("", welcome_msg))
                return "", chatbot
            
            # Special commands
            if message.lower().strip() in ['status', 'info', '/status', '/info', 'help', '/help']:
                if 'help' in message.lower():
                    help_msg = ChatBot._get_help_message()
                    chatbot.append((message, help_msg))
    @staticmethod
    def _get_help_message() -> str:
        """Get help information for users."""
        return """🆘 **AI Assistant Help Guide**

**💬 General Conversation**
- Chat naturally about any topic
- Ask questions about science, technology, history, etc.
- Request explanations or creative help
- Examples: "Tell me about space", "Help me write a story"

**📊 Document Analysis** 
- Upload CSV/XLSX files using the sidebar (📁 button)
- Ask questions about your data once uploaded
- Examples: "What columns are in my data?", "Show statistics"

**🎯 Special Commands**
- `status` or `info` - Check system and document status
- `help` - Show this help message

**💡 Tips**
- Be specific with data questions for better results  
- I can handle both Persian and English
- Use the sidebar for file management
- Clear chat anytime with the 🗑️ button

**🚀 Quick Examples**
- "Hello, how are you?"
- "What's the weather like?" (general chat)
- "Analyze my sales data" (after upload)
- "Explain machine learning"
- "Help me plan a vacation"

Need anything specific? Just ask! 😊"""
                return "", chatbot
            
            # Check if we have documents available
            has_documents = ChatBot._check_collection_exists()
            
            if has_documents:
                # Try document-based response first
                try:
                    # Determine if the question is likely about the documents
                    is_document_related = ChatBot._is_question_document_related(message)
                    
                    if is_document_related:
                        logger.info("Processing as document-related query")
                        response = ChatBot._handle_document_query(message)
                    else:
                        logger.info("Processing as general conversation with document context")
                        response = ChatBot._handle_general_conversation_with_context(message)
                    
                except Exception as e:
                    logger.error(f"Error in document processing: {str(e)}")
                    # Fallback to general conversation
                    response = ChatBot._handle_general_conversation(message)
            else:
                # No documents available, use general conversation mode
                logger.info("No documents available, using general conversation mode")
                response = ChatBot._handle_general_conversation(message)
            
            chatbot.append((message, response))
            return "", chatbot
        
        else:
            logger.info(f"Non-chat functionality: {app_functionality}")
            return "", chatbot
    
    @staticmethod
    def _is_question_document_related(message: str) -> bool:
        """
        Determine if a question is likely about the uploaded documents.
        
        Args:
            message (str): User's message
            
        Returns:
            bool: True if likely document-related, False otherwise
        """
        # Keywords that suggest document-related queries
        document_keywords = [
            'data', 'dataset', 'file', 'document', 'csv', 'excel', 'table', 'column', 'row',
            'analyze', 'analysis', 'statistics', 'stats', 'summary', 'count', 'show me',
            'what is in', 'how many', 'total', 'average', 'mean', 'find', 'search',
            'trend', 'pattern', 'correlation', 'distribution', 'range', 'minimum', 'maximum',
            'داده', 'فایل', 'جدول', 'آنالیز', 'تحلیل', 'آمار'  # Persian keywords
        ]
        
        # Questions that explicitly ask about the documents
        document_phrases = [
            'in my data', 'in the data', 'from the file', 'in this dataset',
            'according to the data', 'based on the file', 'in my document',
            'در داده', 'از فایل', 'در این داده'  # Persian phrases
        ]
        
        message_lower = message.lower()
        
        # Check for explicit document phrases
        if any(phrase in message_lower for phrase in document_phrases):
            return True
        
        # Check for document keywords (require at least 2 for higher confidence)
        keyword_count = sum(1 for keyword in document_keywords if keyword in message_lower)
        
        # Questions starting with 'what', 'how many', 'show', 'find' are likely data queries
        question_starters = ['what', 'how many', 'show', 'find', 'list', 'tell me about']
        starts_with_question = any(message_lower.startswith(starter) for starter in question_starters)
        
        return keyword_count >= 2 or (keyword_count >= 1 and starts_with_question)
    
    @staticmethod
    def _handle_document_query(message: str) -> str:
        """Handle document-specific queries using RAG."""
        try:
            logger.info("Step 1: Generating retrieval query for document search")
            retrieval_query = ChatBot._generate_retrieval_query(message)
            logger.info(f"Generated retrieval query: {retrieval_query}")
            
            logger.info("Step 2: Generating embeddings for retrieval")
            query_embeddings = APPCFG.embeddings_client.embed_query(retrieval_query)
            
            logger.info("Step 3: Retrieving relevant documents from ChromaDB")
            vectordb = APPCFG.chroma_client.get_collection(name=APPCFG.collection_name)
            results = vectordb.query(
                query_embeddings=[query_embeddings],
                n_results=APPCFG.top_k
            )
            logger.info(f"Retrieved {len(results['documents'][0])} relevant documents")
            
            # Check relevance of retrieved documents
            formatted_docs = ChatBot._format_retrieved_documents(results)
            
            # Generate response using retrieved context
            logger.info("Step 4: Generating response with document context")
            response = ChatBot._generate_document_response(message, formatted_docs)
            
            # Log statistics
            ChatBot._log_statistics(results, response)
            return response
            
        except Exception as e:
            logger.error(f"Error in document query processing: {str(e)}")
            return f"I encountered an error while searching the documents. Let me try to help you in general: {ChatBot._handle_general_conversation(message)}"
    
    @staticmethod
    def _handle_general_conversation_with_context(message: str) -> str:
        """Handle general conversation but mention document availability."""
        try:
            # Get basic info about available documents
            collection_summary = ChatBot._get_brief_collection_summary()
            
            general_prompt = f"""You are a helpful AI assistant. The user has uploaded documents ({collection_summary}) but is asking a general question that doesn't seem to be specifically about analyzing those documents.

User's question: {message}

Instructions:
- Answer the user's question normally as you would in any conversation
- Be helpful, friendly, and informative
- You can mention that you also have access to their uploaded documents if relevant
- Don't force document analysis unless the user specifically asks for it
- Keep the response natural and conversational

Response:"""
            
            messages = [
                {"role": "system", "content": "You are a helpful, friendly AI assistant capable of both general conversation and document analysis."},
                {"role": "user", "content": general_prompt}
            ]
            
            llm_response = APPCFG.azure_openai_client.invoke(messages)
            return llm_response.content
            
        except Exception as e:
            logger.error(f"Error in general conversation with context: {str(e)}")
            return ChatBot._handle_general_conversation(message)
    
    @staticmethod
    def _handle_general_conversation(message: str) -> str:
        """Handle general conversation without document context."""
        try:
            general_prompt = f"""You are a helpful, friendly AI assistant. Have a natural conversation with the user.

User's message: {message}

Instructions:
- Be helpful, conversational, and engaging
- Answer questions to the best of your ability
- If the user asks about documents or data analysis, mention that they can upload CSV/XLSX files for analysis
- Keep responses natural and appropriately detailed
- You can discuss any topic the user is interested in

Response:"""
            
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant. You're friendly, knowledgeable, and enjoy conversations on a wide range of topics. You can also help analyze documents when users upload them."},
                {"role": "user", "content": general_prompt}
            ]
            
            llm_response = APPCFG.azure_openai_client.invoke(messages)
            return llm_response.content
            
        except Exception as e:
            logger.error(f"Error in general conversation: {str(e)}")
            return "I apologize, but I'm having trouble processing your message right now. Could you please try again?"
    
    @staticmethod
    def _check_collection_exists() -> bool:
        """Check if the ChromaDB collection exists and has documents."""
        try:
            collection = APPCFG.chroma_client.get_collection(name=APPCFG.collection_name)
            count = collection.count()
            logger.info(f"Collection '{APPCFG.collection_name}' found with {count} documents")
            return count > 0
        except Exception as e:
            logger.info(f"No collection found: {str(e)}")
            return False
    
    @staticmethod
    def _get_brief_collection_summary() -> str:
        """Get a brief summary of the collection for context."""
        try:
            collection = APPCFG.chroma_client.get_collection(name=APPCFG.collection_name)
            count = collection.count()
            
            # Get a sample to understand the data source
            sample = collection.query(query_embeddings=[[0.0] * 1536], n_results=1)
            source = "unknown source"
            if sample['metadatas'] and sample['metadatas'][0]:
                source = sample['metadatas'][0][0].get('source', 'unknown source')
            
            return f"{count} documents from {source}"
        except:
            return "uploaded documents"
    
    @staticmethod
    def _get_collection_status() -> str:
        """Get detailed status of the current collection."""
        try:
            collection = APPCFG.chroma_client.get_collection(name=APPCFG.collection_name)
            count = collection.count()
            
            # Get sample documents to understand the data
            sample = collection.query(
                query_embeddings=[[0.0] * 1536],
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

**📁 Collection Info:**
- **Name**: `{APPCFG.collection_name}`
- **Total Documents**: **{count:,}**
- **Data Sources**: **{', '.join(sources)}**

**📋 Sample Data Preview:**
```
{sample_content[0] if sample_content else 'No data available'}
```

**🎯 Ready for:**
- 📊 Document analysis and data queries
- 💬 General conversations and questions
- 🔍 Information extraction and insights

**💡 Example Questions:**
- "What columns are in my data?"
- "Show me some statistics"
- "Hello, how are you?" (general chat)
- "Explain machine learning to me"
"""
            return status_msg
            
        except Exception as e:
            return f"""📭 **No Documents Available**

**Status**: No collection found

**🎯 What you can do:**
- 📤 Upload CSV/XLSX files using the sidebar
- 💬 Chat with me about any topic you're interested in
- ❓ Ask me questions - I'm here to help!

**💡 I can discuss:**
- Technology, science, history
- Help with explanations and learning
- Creative writing and brainstorming
- Data analysis (once you upload files)

**Error details**: {str(e)}"""
    
    @staticmethod
    def _generate_retrieval_query(user_message: str) -> str:
        """Generate an optimized query for document retrieval."""
        retrieval_prompt = f"""Based on the user's question about their uploaded documents, generate a focused search query that will help find the most relevant information from the database.

User's question: {user_message}

Generate a concise search query that captures the key concepts and entities the user is asking about. Focus on the main terms that would appear in relevant documents.

Search query:"""
        
        messages = [
            {"role": "system", "content": "You are an expert at generating search queries for document retrieval. Create focused, effective queries."},
            {"role": "user", "content": retrieval_prompt}
        ]
        
        try:
            llm_response = APPCFG.azure_openai_client.invoke(messages)
            retrieval_query = llm_response.content.strip()
            
            if len(retrieval_query) < 3:
                retrieval_query = user_message
                
            return retrieval_query
        except Exception as e:
            logger.warning(f"Failed to generate retrieval query: {str(e)}")
            return user_message
    
    @staticmethod
    def _format_retrieved_documents(results) -> str:
        """Format the retrieved documents for LLM processing."""
        if not results['documents'] or not results['documents'][0]:
            return "No relevant documents found."
        
        formatted_docs = "**Retrieved Documents:**\n\n"
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0], 
            results['metadatas'][0], 
            results['distances'][0]
        )):
            relevance = max(0, 1 - distance)  # Convert distance to relevance score
            formatted_docs += f"**Document {i+1}** (Source: {metadata.get('source', 'Unknown')}, Relevance: {relevance:.3f})\n"
            formatted_docs += f"{doc}\n\n"
        
        return formatted_docs
    
    @staticmethod
    def _generate_document_response(user_message: str, context_docs: str) -> str:
        """Generate response using retrieved documents."""
        prompt = f"""You are an expert data analyst helping a user understand their uploaded documents. Based on the user's question and the retrieved relevant documents, provide a comprehensive and helpful answer.

User's question: {user_message}

{context_docs}

Instructions:
- Analyze the retrieved documents to answer the user's question
- Be specific and cite relevant details from the documents
- If the documents don't fully answer the question, explain what you can determine and what might be missing
- Present information clearly with appropriate formatting
- If you find patterns or insights, highlight them
- Be helpful and thorough in your analysis

Answer:"""
        
        messages = [
            {"role": "system", "content": "You are an expert data analyst and helpful assistant. Analyze documents thoroughly and provide clear, actionable insights."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            llm_response = APPCFG.azure_openai_client.invoke(messages)
            return llm_response.content
        except Exception as e:
            logger.error(f"Failed to generate document response: {str(e)}")
            return f"I found relevant information in your documents, but encountered an error generating the response. The key content includes: {context_docs[:200]}..."
    
    @staticmethod
    def _log_statistics(results, response):
        """Log statistics about the retrieval and response."""
        try:
            num_docs_retrieved = len(results['documents'][0]) if results['documents'] else 0
            avg_distance = sum(results['distances'][0]) / len(results['distances'][0]) if results['distances'] and results['distances'][0] else 0
            response_length = len(response.split())
            
            logger.info(f"📊 Query statistics - Documents retrieved: {num_docs_retrieved}, "
                       f"Average relevance: {1-avg_distance:.3f}, "
                       f"Response length: {response_length} words")
        except Exception as e:
            logger.warning(f"Failed to log statistics: {str(e)}")