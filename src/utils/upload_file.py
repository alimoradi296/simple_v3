import os
import logging
from typing import List, Tuple
from utils.load_config import LoadConfig
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

APPCFG = LoadConfig()


class ProcessFiles:
    """
    A unified class to process uploaded files and convert them to embeddings in ChromaDB.
    Supports both CSV and XLSX files.
    """
    def __init__(self, files_dir: List, chatbot: List) -> None:
        """
        Initialize the ProcessFiles instance.

        Args:
            files_dir (List): A list containing the file paths of uploaded files.
            chatbot (List): A list representing the chatbot's conversation history.
        """
        self.files_dir = files_dir
        self.chatbot = chatbot
        logger.info(f"Initializing ProcessFiles with {len(self.files_dir)} files")

    def _check_existing_collection(self) -> bool:
        """Check if collection already exists and handle replacement automatically."""
        try:
            collection = APPCFG.chroma_client.get_collection(name=APPCFG.collection_name)
            count = collection.count()
            if count > 0:
                logger.warning(f"Collection '{APPCFG.collection_name}' already exists with {count} documents")
                
                # Automatically replace existing collection without user prompt
                logger.info("🔄 Replacing existing collection with new data")
                APPCFG.chroma_client.delete_collection(name=APPCFG.collection_name)
                
                # Add a message to inform the user
                self.chatbot.append((" ", f"""🔄 **Replacing Existing Data**
                
Found {count} existing documents in the system. They will be replaced with your new files.
                
**Previous Collection**: {count} documents
**Status**: Clearing for new upload...
"""))
                
                return False  # Proceed with upload after deletion
        except Exception:
            logger.info("No existing collection found, will create new one")
        return False

    def _process_uploaded_files(self) -> Tuple:
        """
        Process uploaded files and convert them to embeddings in ChromaDB with progress tracking.

        Returns:
            Tuple[str, List]: A tuple containing an empty string and the updated chatbot conversation list.
        """
        logger.info("Starting file processing pipeline with batch embeddings")
        
        # Check for existing collection and handle replacement
        existing_collection_found = self._check_existing_collection()
        if existing_collection_found:
            # This case should not occur now since we auto-delete in _check_existing_collection
            error_msg = f"❌ Upload cancelled due to existing data."
            self.chatbot.append((" ", error_msg))
            logger.warning("Upload cancelled due to existing collection")
            return "", self.chatbot

        # Show initial progress
        self.chatbot.append((" ", f"🚀 **Processing {len(self.files_dir)} file(s)...**\n\nPlease wait while we process your files. This may take a few minutes for large files."))

        # Process each file
        all_docs = []
        all_metadatas = []
        all_ids = []
        all_embeddings = []
        
        total_rows_processed = 0
        
        for file_index, file_dir in enumerate(self.files_dir):
            file_progress = f"{file_index + 1}/{len(self.files_dir)}"
            logger.info(f"Processing file {file_progress}: {os.path.basename(file_dir)}")
            
            # Update progress in chat
            self.chatbot.append((" ", f"📄 **Processing file {file_progress}**: `{os.path.basename(file_dir)}`"))
            
            try:
                # Load the file
                df, file_name = self._load_dataframe(file_dir)
                logger.info(f"Loaded {file_name} with {len(df)} rows and {len(df.columns)} columns")
                
                # Update chat with file details
                self.chatbot.append((" ", f"✅ **Loaded**: {len(df)} rows, {len(df.columns)} columns\n🔄 **Generating embeddings...**"))
                
                # Generate embeddings for this file (now uses batch processing)
                docs, metadatas, ids, embeddings = self._prepare_data_for_embedding(df, file_name, file_index)
                
                # Add to combined lists
                all_docs.extend(docs)
                all_metadatas.extend(metadatas)
                all_ids.extend(ids)
                all_embeddings.extend(embeddings)
                
                total_rows_processed += len(df)
                
                logger.info(f"Generated {len(embeddings)} embeddings for {file_name}")
                self.chatbot.append((" ", f"✅ **Completed**: {len(embeddings)} embeddings generated for `{file_name}`"))
                
            except Exception as e:
                logger.error(f"Error processing file {os.path.basename(file_dir)}: {str(e)}")
                self.chatbot.append((" ", f"❌ **Error processing** `{os.path.basename(file_dir)}`: {str(e)}"))
                continue
        
        if all_docs:
            # Update chat before storing
            self.chatbot.append((" ", f"💾 **Storing {len(all_docs)} documents in database...**\n\nThis may take a moment for large datasets."))
            
            # Store all embeddings in ChromaDB
            logger.info(f"Storing {len(all_docs)} total documents in ChromaDB")
            self._store_in_chromadb(all_docs, all_metadatas, all_ids, all_embeddings)
            
            # Final success message with statistics
            success_msg = f"""✅ **Upload Complete!**

📊 **Summary:**
- **Files Processed**: {len(self.files_dir)}
- **Total Documents**: {len(all_docs)}
- **Total Rows**: {total_rows_processed}

🎉 **Ready to Chat!** 
Switch to 'Chat' mode and start asking questions about your data!

**Examples:**
- "What are the main columns in my data?"
- "Show me some statistics"
- "What patterns can you find?"
"""
            
            self.chatbot.append((" ", success_msg))
            logger.info("File processing completed successfully")
        else:
            error_msg = """❌ **No Files Processed**

None of the uploaded files could be processed successfully. 

**Please check:**
- File formats (CSV, XLSX supported)
- File integrity 
- File permissions

Try uploading different files or check the console logs for detailed error information."""
            
            self.chatbot.append((" ", error_msg))
            logger.error("No files were processed successfully")
        
        return "", self.chatbot

    def _load_dataframe(self, file_dir: str) -> Tuple[pd.DataFrame, str]:
        """
        Load a DataFrame from CSV or XLSX file.
        
        Args:
            file_dir (str): Path to the file
            
        Returns:
            Tuple[pd.DataFrame, str]: DataFrame and filename without extension
        """
        file_names_with_extensions = os.path.basename(file_dir)
        file_name, file_extension = os.path.splitext(file_names_with_extensions)
        
        logger.info(f"Loading file: {file_names_with_extensions}")
        
        if file_extension.lower() == ".csv":
            df = pd.read_csv(file_dir)
        elif file_extension.lower() == ".xlsx":
            df = pd.read_excel(file_dir)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        return df, file_name

    def _prepare_data_for_embedding(self, df: pd.DataFrame, file_name: str, file_index: int) -> Tuple[List, List, List, List]:
        """
        Prepare data for embedding generation using batch processing.
        
        Args:
            df (pd.DataFrame): The DataFrame to process
            file_name (str): Name of the source file
            file_index (int): Index of the file being processed
            
        Returns:
            Tuple[List, List, List, List]: Documents, metadatas, ids, and embeddings
        """
        docs = []
        metadatas = []
        ids = []
        
        logger.info(f"Preparing {len(df)} rows for batch embedding in {file_name}")
        
        # Prepare all documents first
        for row_index, row in df.iterrows():
            # Create a structured document from the row
            doc_content = self._format_row_content(df.columns, row)
            
            # Store the data (except embeddings)
            docs.append(doc_content)
            metadatas.append({
                "source": file_name,
                "row_index": row_index,
                "file_index": file_index,
                "total_rows": len(df)
            })
            ids.append(f"{file_name}_row_{row_index}")
        
        # Generate embeddings in batches
        logger.info(f"Generating embeddings in batches for {len(docs)} documents")
        embeddings = self._generate_batch_embeddings(docs)
        
        logger.info(f"Successfully generated {len(embeddings)} embeddings for {file_name}")
        return docs, metadatas, ids, embeddings

    def _generate_batch_embeddings(self, documents: List[str]) -> List[List[float]]:
        """
        Generate embeddings for documents in batches for efficiency.
        
        Args:
            documents (List[str]): List of document texts
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        batch_size = APPCFG.embedding_batch_size
        all_embeddings = []
        total_batches = (len(documents) + batch_size - 1) // batch_size
        
        logger.info(f"Processing {len(documents)} documents in {total_batches} batches of {batch_size}")
        
        for i in range(0, len(documents), batch_size):
            batch_end = min(i + batch_size, len(documents))
            batch = documents[i:batch_end]
            batch_num = i // batch_size + 1
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} documents)")
            
            try:
                # Use the embeddings client to generate batch embeddings
                batch_embeddings = APPCFG.embeddings_client.embed_documents(batch)
                all_embeddings.extend(batch_embeddings)
                
                logger.info(f"✅ Successfully processed batch {batch_num}/{total_batches}")
                
                # Add a small delay between batches to avoid rate limiting
                if i + batch_size < len(documents):
                    import time
                    time.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"❌ Failed to process batch {batch_num}: {str(e)}")
                
                # Fallback to individual processing for this batch
                logger.info("🔄 Falling back to individual embedding generation for failed batch")
                for j, doc in enumerate(batch):
                    try:
                        embedding = APPCFG.embeddings_client.embed_query(doc)
                        all_embeddings.append(embedding)
                        if (j + 1) % 10 == 0:
                            logger.info(f"   Individual processing: {j + 1}/{len(batch)} documents")
                    except Exception as doc_error:
                        logger.error(f"❌ Failed to embed individual document: {str(doc_error)}")
                        # Use zero vector as fallback (this should rarely happen)
                        all_embeddings.append([0.0] * 1536)
        
        logger.info(f"✅ Batch embedding generation completed: {len(all_embeddings)} embeddings generated")
        return all_embeddings

    def _format_row_content(self, columns: List, row: pd.Series) -> str:
        """
        Format a DataFrame row into a structured text document.
        
        Args:
            columns (List): Column names
            row (pd.Series): The row data
            
        Returns:
            str: Formatted document content
        """
        content_parts = []
        for col in columns:
            value = str(row[col]) if pd.notna(row[col]) else "N/A"
            content_parts.append(f"{col}: {value}")
        
        return ", ".join(content_parts)

    def _store_in_chromadb(self, docs: List, metadatas: List, ids: List, embeddings: List):
        """
        Store the processed data in ChromaDB with optimized batch processing.
        
        Args:
            docs (List): List of document contents
            metadatas (List): List of metadata dictionaries
            ids (List): List of document IDs
            embeddings (List): List of embeddings
        """
        try:
            logger.info(f"Creating ChromaDB collection: {APPCFG.collection_name}")
            collection = APPCFG.chroma_client.create_collection(name=APPCFG.collection_name)
            
            # Use configurable batch size for storage
            batch_size = APPCFG.storage_batch_size
            total_batches = (len(docs) + batch_size - 1) // batch_size
            
            logger.info(f"Storing {len(docs)} documents in {total_batches} batches of up to {batch_size}")
            
            for i in range(0, len(docs), batch_size):
                batch_end = min(i + batch_size, len(docs))
                batch_num = i // batch_size + 1
                
                logger.info(f"📦 Storing batch {batch_num}/{total_batches} (documents {i+1}-{batch_end} of {len(docs)})")
                
                collection.add(
                    documents=docs[i:batch_end],
                    metadatas=metadatas[i:batch_end],
                    embeddings=embeddings[i:batch_end],
                    ids=ids[i:batch_end]
                )
                
                # Log progress with percentage
                progress = (batch_end / len(docs)) * 100
                logger.info(f"📊 Storage progress: {progress:.1f}% complete")
            
            logger.info(f"✅ Successfully stored all {len(docs)} documents in ChromaDB")
            
        except Exception as e:
            logger.error(f"❌ Failed to store data in ChromaDB: {str(e)}")
            raisesize = 500
            total_batches = (len(docs) + batch_size - 1) // batch_size
            
            logger.info(f"Storing {len(docs)} documents in {total_batches} batches of up to {batch_size}")
            
            for i in range(0, len(docs), batch_size):
                batch_end = min(i + batch_size, len(docs))
                batch_num = i // batch_size + 1
                
                logger.info(f"Storing batch {batch_num}/{total_batches} (documents {i+1}-{batch_end} of {len(docs)})")
                
                collection.add(
                    documents=docs[i:batch_end],
                    metadatas=metadatas[i:batch_end],
                    embeddings=embeddings[i:batch_end],
                    ids=ids[i:batch_end]
                )
                
                # Log progress with percentage
                progress = (batch_end / len(docs)) * 100
                logger.info(f"Storage progress: {progress:.1f}% complete")
            
            logger.info(f"✅ Successfully stored all {len(docs)} documents in ChromaDB")
            
        except Exception as e:
            logger.error(f"❌ Failed to store data in ChromaDB: {str(e)}")
            raise

    def _validate_db(self):
        """Validate that the ChromaDB collection was created successfully with enhanced reporting."""
        try:
            collection = APPCFG.chroma_client.get_collection(name=APPCFG.collection_name)
            count = collection.count()
            logger.info(f"✅ Validation successful: ChromaDB collection contains {count} documents")
            
            # Get sample documents to verify structure and show user what was stored
            sample_size = min(3, count)
            sample = collection.query(
                query_embeddings=[[0.0] * 1536],  # Using zero vector for sampling
                n_results=sample_size
            )
            
            if sample['documents'] and sample['metadatas']:
                logger.info("Sample documents preview:")
                for i, (doc, metadata) in enumerate(zip(sample['documents'][0], sample['metadatas'][0])):
                    logger.info(f"  {i+1}. Source: {metadata.get('source', 'Unknown')}")
                    logger.info(f"     Preview: {doc[:100]}...")
                    
            # Calculate and report some statistics
            sources = set()
            if sample['metadatas']:
                for metadata in sample['metadatas'][0]:
                    sources.add(metadata.get('source', 'Unknown'))
            
            logger.info(f"📊 Collection Stats: {count} total documents from {len(sources)} source(s): {', '.join(sources)}")
                
        except Exception as e:
            logger.error(f"❌ Validation failed: {str(e)}")
            raise

    def run(self):
        """
        Execute the complete file processing pipeline with performance monitoring.

        Returns:
            Tuple[str, List]: A tuple containing an empty string and the updated chatbot conversation list.
        """
        import time
        start_time = time.time()
        
        logger.info("🚀 Starting file processing pipeline with batch optimization")
        
        try:
            input_txt, chatbot = self._process_uploaded_files()
            
            # Only validate if processing was successful
            if any("✅ **Upload Complete!**" in str(msg) for msg in chatbot if len(msg) > 1):
                self._validate_db()
                
                # Calculate and log performance metrics
                end_time = time.time()
                total_time = end_time - start_time
                
                # Get final collection stats
                try:
                    collection = APPCFG.chroma_client.get_collection(name=APPCFG.collection_name)
                    total_docs = collection.count()
                    
                    # Calculate performance metrics
                    docs_per_second = total_docs / total_time if total_time > 0 else 0
                    
                    performance_msg = f"""📊 **Performance Summary**
- **Total Processing Time**: {total_time:.1f} seconds
- **Documents Processed**: {total_docs}
- **Processing Rate**: {docs_per_second:.1f} docs/second
- **Batch Sizes Used**: Embedding={APPCFG.embedding_batch_size}, Storage={APPCFG.storage_batch_size}"""
                    
                    logger.info(f"Performance: {total_time:.1f}s total, {docs_per_second:.1f} docs/sec")
                    self.chatbot.append((" ", performance_msg))
                    
                except Exception as perf_error:
                    logger.warning(f"Could not calculate performance metrics: {str(perf_error)}")
                
                logger.info("✅ File processing pipeline completed successfully")
            else:
                logger.warning("⚠️ File processing pipeline completed with errors")
                
        except Exception as e:
            logger.error(f"❌ File processing pipeline failed: {str(e)}")
            self.chatbot.append((" ", f"❌ **Processing Failed**: {str(e)}"))
            return "", self.chatbot
            
        return input_txt, chatbot


class UploadFile:
    """
    Controller class for running the file processing pipeline.
    """
    @staticmethod
    def run_pipeline(files_dir: List, chatbot: List, chatbot_functionality: str):
        """
        Run the file processing pipeline.

        Args:
            files_dir (List): List of uploaded file paths
            chatbot (List): Current chatbot conversation state
            chatbot_functionality (str): The functionality mode

        Returns:
            Tuple: Updated state after processing
        """
        logger.info(f"Starting upload pipeline with functionality: {chatbot_functionality}")
        
        if chatbot_functionality == "Process files":
            pipeline_instance = ProcessFiles(files_dir=files_dir, chatbot=chatbot)
            input_txt, chatbot = pipeline_instance.run()
            return input_txt, chatbot
        elif chatbot_functionality == "Chat":
            # User tried to upload files while in Chat mode
            error_msg = "❌ **Upload Error**: You tried to upload files while in 'Chat' mode. Please:\n\n1. Switch to 'Process files' mode first\n2. Then upload your files\n3. Switch back to 'Chat' mode to ask questions"
            logger.warning("Files uploaded in Chat mode - redirecting user")
            chatbot.append((" ", error_msg))
            return "", chatbot
        else:
            logger.warning(f"Unsupported functionality: {chatbot_functionality}")
            chatbot.append((" ", f"❌ Functionality '{chatbot_functionality}' is not supported. Please use 'Process files' or 'Chat' mode."))
            return "", chatbot