import gradio as gr
import logging
from utils.upload_file import UploadFile
from utils.chatbot import ChatBot
from utils.ui_settings import UISettings

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global state for sidebar
show_sidebar = gr.State(False)

with gr.Blocks(title="AI Document Assistant", css="""
    .sidebar {
        background: #f8f9fa;
        border-right: 1px solid #dee2e6;
        padding: 1rem;
        height: 100vh;
        overflow-y: auto;
    }
    
    .main-content {
        padding: 1rem;
    }
    
    .upload-area {
        border: 2px dashed #dee2e6;
        border-radius: 8px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        background: #fff;
    }
    
    .file-list {
        background: #fff;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        max-height: 300px;
        overflow-y: auto;
    }
    
    .chatbot-container {
        border: 1px solid #dee2e6;
        border-radius: 12px;
        overflow: hidden;
    }
    
    .status-card {
        background: #e7f3ff;
        border: 1px solid #b3d7ff;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .toggle-btn {
        position: fixed;
        top: 20px;
        left: 20px;
        z-index: 1000;
        background: #007bff;
        color: white;
        border: none;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        font-size: 18px;
        cursor: pointer;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
""") as demo:
    
    logger.info("Initializing AI Document Assistant interface")
    
    # Sidebar toggle button
    with gr.Row():
        with gr.Column(scale=0, min_width=50):
            sidebar_toggle = gr.Button("📁", elem_classes="toggle-btn", size="sm")
    
    with gr.Row():
        # Sidebar
        with gr.Column(scale=1, visible=False, elem_classes="sidebar") as sidebar:
            with gr.Accordion("📁 File Management", open=True):
                gr.Markdown("### Upload Documents")
                
                upload_btn = gr.UploadButton(
                    "📤 Upload CSV/XLSX Files", 
                    file_types=['.csv', '.xlsx'], 
                    file_count="multiple",
                    variant="primary",
                    size="lg"
                )
                
                # File status display
                gr.Markdown("### Document Status")
                file_status = gr.Markdown("📭 **No documents uploaded**\n\nUpload CSV or XLSX files to get started!")
                
                # Collection management
                gr.Markdown("### Collection Management")
                with gr.Row():
                    refresh_btn = gr.Button("🔄 Refresh Status", variant="secondary", size="sm")
                    clear_btn = gr.Button("🗑️ Clear Collection", variant="secondary", size="sm")
                
                # Upload progress
                gr.Markdown("### Upload Progress")
                upload_progress = gr.Markdown("Ready to upload files...")
        
        # Main chat area
        with gr.Column(scale=4, elem_classes="main-content"):
            # Header
            gr.Markdown("""
            # 🤖 AI Document Assistant
            
            **Your intelligent assistant for document analysis and general conversations**
            
            💡 **I can help you with:**
            - 📊 Analyzing your uploaded CSV/XLSX documents
            - 💬 General conversations and questions
            - 📈 Data insights and statistics
            - 🔍 Information extraction from your files
            """)
            
            # Main chatbot
            chatbot = gr.Chatbot(
                [],
                elem_id="chatbot",
                bubble_full_width=False,
                height=500,
                label="💬 Chat with your AI Assistant",
                elem_classes="chatbot-container"
            )
            
            # Like/dislike feedback
            chatbot.like(UISettings.feedback, None, None)
            
            # Input area
            with gr.Row():
                input_txt = gr.Textbox(
                    lines=2,
                    scale=8,
                    placeholder="Ask me anything! I can help with your documents or answer general questions...",
                    container=False,
                    label="Your Message"
                )
                
                with gr.Column(scale=1):
                    send_btn = gr.Button("🚀 Send", variant="primary", size="lg")
                    clear_chat_btn = gr.Button("🗑️ Clear Chat", variant="secondary", size="sm")
    
    # State management
    sidebar_state = gr.State(False)
    
    # Event handlers
    def toggle_sidebar(current_state):
        """Toggle sidebar visibility."""
        new_state = not current_state
        return gr.Column(visible=new_state), new_state
    
    def update_file_status():
        """Update the file status in sidebar."""
        try:
            from utils.load_config import LoadConfig
            APPCFG = LoadConfig()
            
            try:
                collection = APPCFG.chroma_client.get_collection(name=APPCFG.collection_name)
                count = collection.count()
                
                # Get sample to understand data
                sample = collection.query(query_embeddings=[[0.0] * 1536], n_results=1)
                source = "Unknown"
                if sample['metadatas'] and sample['metadatas'][0]:
                    source = sample['metadatas'][0][0].get('source', 'Unknown')
                
                status = f"""✅ **Documents Loaded**
                
**📊 Collection Stats:**
- **Documents**: {count:,}
- **Source**: `{source}`
- **Status**: Ready for queries

🎯 **Quick Actions:**
- Switch to chat to ask questions
- Upload new files to replace current data
"""
            except:
                status = """📭 **No Documents Found**

**📝 Upload Instructions:**
1. Click '📤 Upload CSV/XLSX Files'
2. Select your files
3. Wait for processing
4. Start chatting!

**💡 Tip:** I can chat about general topics too!"""
                
            return status
        except Exception as e:
            return f"❌ Error checking status: {str(e)}"
    
    def clear_collection():
        """Clear the current collection."""
        try:
            from utils.load_config import LoadConfig
            APPCFG = LoadConfig()
            
            try:
                APPCFG.chroma_client.delete_collection(name=APPCFG.collection_name)
                logger.info("Collection cleared successfully")
                return "✅ Collection cleared successfully!", update_file_status()
            except:
                return "ℹ️ No collection to clear.", update_file_status()
        except Exception as e:
            return f"❌ Error clearing collection: {str(e)}", update_file_status()
    
    def handle_file_upload(files, progress_display):
        """Handle file upload and update progress."""
        if not files:
            return "❌ No files selected", update_file_status()
        
        try:
            # Create a temporary chatbot list for progress tracking
            temp_chatbot = []
            
            # Process files
            input_txt, temp_chatbot = UploadFile.run_pipeline(files, temp_chatbot, "Process files")
            
            # Extract progress messages
            progress_msgs = []
            for msg in temp_chatbot:
                if len(msg) > 1:
                    progress_msgs.append(msg[1])
            
            # Combine all progress messages
            final_progress = "\n\n".join(progress_msgs) if progress_msgs else "Upload completed"
            
            # Check if upload was successful
            if any("✅ **Upload Complete!**" in str(msg) for msg in progress_msgs):
                final_status = "✅ Upload successful! Files are ready for queries."
            else:
                final_status = "⚠️ Upload completed with issues. Check the progress above."
            
            return final_progress + f"\n\n{final_status}", update_file_status()
            
        except Exception as e:
            error_msg = f"❌ Upload failed: {str(e)}"
            logger.error(error_msg)
            return error_msg, update_file_status()
    
    def enhanced_chat_respond(history, message):
        """Enhanced chat response that handles both document and general queries."""
        # Use the enhanced ChatBot.respond method
        _, updated_history = ChatBot.respond(history, message, "Enhanced Chat", "Chat")
        return updated_history, ""
    
    # Wire up event handlers
    sidebar_toggle.click(
        fn=toggle_sidebar,
        inputs=[sidebar_state],
        outputs=[sidebar, sidebar_state]
    )
    
    refresh_btn.click(
        fn=update_file_status,
        outputs=[file_status]
    )
    
    clear_btn.click(
        fn=clear_collection,
        outputs=[upload_progress, file_status]
    )
    
    upload_btn.upload(
        fn=handle_file_upload,
        inputs=[upload_btn, upload_progress],
        outputs=[upload_progress, file_status]
    )
    
    # Chat handlers
    def submit_and_clear():
        return gr.Textbox(value="", interactive=True)
    
    input_txt.submit(
        fn=enhanced_chat_respond,
        inputs=[chatbot, input_txt],
        outputs=[chatbot, input_txt]
    ).then(
        submit_and_clear,
        outputs=[input_txt]
    )
    
    send_btn.click(
        fn=enhanced_chat_respond,
        inputs=[chatbot, input_txt],
        outputs=[chatbot, input_txt]
    ).then(
        submit_and_clear,
        outputs=[input_txt]
    )
    
    clear_chat_btn.click(
        lambda: [],
        outputs=[chatbot]
    )
    
    # Initialize with welcome message
    def get_initial_chat():
        """Get initial chat with welcome message."""
        try:
            from utils.chatbot import ChatBot
            return [("", ChatBot.get_welcome_message())]
        except Exception as e:
            logger.warning(f"Could not load welcome message: {e}")
            return [("", "👋 Welcome! I'm your AI assistant. How can I help you today?")]
    
    # Initialize file status on load and set welcome message
    demo.load(
        fn=lambda: (update_file_status(), get_initial_chat()),
        outputs=[file_status, chatbot]
    )

if __name__ == "__main__":
    logger.info("Starting AI Document Assistant")
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
    logger.info("AI Document Assistant stopped")