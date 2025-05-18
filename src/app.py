import gradio as gr
import logging
from utils.upload_file import UploadFile
from utils.chatbot import ChatBot
from utils.ui_settings import UISettings

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

with gr.Blocks(title="Unified RAG Document Q&A System") as demo:
    with gr.Tabs():
        with gr.TabItem("📄 RAG Document Q&A"):
            logger.info("Initializing Gradio interface")
            
            ##############
            # Header with instructions
            ##############
            gr.Markdown("""
            # 🤖 Unified RAG Document Q&A System
            
            **How to use:**
            1. **Upload Files**: Switch to "Process files" mode and upload your CSV/XLSX files
            2. **Chat**: Switch back to "Chat" mode and ask questions about your uploaded data
            
            **Features:**
            - Intelligent document retrieval using embeddings
            - LLM-guided search query optimization  
            - Comprehensive logging for transparency
            - Support for CSV and XLSX files
            """)
            
            ##############
            # First ROW: Chatbot
            ##############
            with gr.Row() as row_one:
                chatbot = gr.Chatbot(
                    [],
                    elem_id="chatbot",
                    bubble_full_width=False,
                    height=500,
                    avatar_images=("images/AI_RT.png", "images/openai.png"),
                    label="💬 Chat with your documents"
                )
                # Adding like/dislike feedback
                chatbot.like(UISettings.feedback, None, None)
            
            ##############
            # Second ROW: Input
            ##############
            with gr.Row():
                input_txt = gr.Textbox(
                    lines=4,
                    scale=8,
                    placeholder="Ask questions about your uploaded documents (e.g., 'What are the key trends in the data?', 'Show me statistics about...')",
                    container=False,
                    label="Your Question"
                )
            
            ##############
            # Third ROW: Controls
            ##############
            with gr.Row() as row_two:
                with gr.Column(scale=2):
                    text_submit_btn = gr.Button(value="🚀 Submit Question", variant="primary")
                    
                with gr.Column(scale=2):
                    upload_btn = gr.UploadButton(
                        "📁 Upload CSV/XLSX Files", 
                        file_types=['.csv', '.xlsx'], 
                        file_count="multiple",
                        variant="secondary"
                    )
                
                with gr.Column(scale=2):
                    app_functionality = gr.Dropdown(
                        label="🔧 Mode", 
                        choices=["Chat", "Process files"], 
                        value="Chat",
                        info="Switch between chatting and uploading files"
                    )
                
                with gr.Column(scale=1):
                    clear_button = gr.ClearButton([input_txt, chatbot], value="🗑️ Clear")
            
            ##############
            # Fourth ROW: Status and Information
            ##############
            with gr.Row():
                gr.Markdown("""
                ### 📊 System Status
                - **Current Mode**: Check the mode dropdown above
                - **Ready to Chat**: Upload files first, then switch to Chat mode
                - **Logs**: Check the console for detailed processing logs
                """)
            
            ##############
            # Event Handlers
            ##############
            
            # File upload handler
            file_msg = upload_btn.upload(
                fn=UploadFile.run_pipeline, 
                inputs=[upload_btn, chatbot, app_functionality], 
                outputs=[input_txt, chatbot], 
                queue=False
            )

            # Text input submission handlers
            def submit_and_clear():
                """Submit text and clear input, then make input interactive again."""
                return gr.Textbox(interactive=True)

            # Text submission via Enter key
            txt_msg = input_txt.submit(
                fn=ChatBot.respond,
                inputs=[chatbot, input_txt, gr.State("RAG Document Q&A"), app_functionality],
                outputs=[input_txt, chatbot],
                queue=False
            ).then(
                submit_and_clear,
                None, 
                [input_txt], 
                queue=False
            )

            # Text submission via button click
            btn_msg = text_submit_btn.click(
                fn=ChatBot.respond,
                inputs=[chatbot, input_txt, gr.State("RAG Document Q&A"), app_functionality],
                outputs=[input_txt, chatbot],
                queue=False
            ).then(
                submit_and_clear,
                None, 
                [input_txt], 
                queue=False
            )

            ##############
            # Mode change handler
            ##############
            def update_interface_based_on_mode(mode):
                """Update interface elements based on selected mode."""
                if mode == "Process files":
                    return {
                        input_txt: gr.Textbox(
                            placeholder="Upload CSV/XLSX files using the upload button above, then switch back to Chat mode",
                            interactive=False
                        ),
                        text_submit_btn: gr.Button(value="🚀 Submit Question", interactive=False)
                    }
                else:  # Chat mode
                    return {
                        input_txt: gr.Textbox(
                            placeholder="Ask questions about your uploaded documents (e.g., 'What are the key trends in the data?', 'Show me statistics about...')",
                            interactive=True
                        ),
                        text_submit_btn: gr.Button(value="🚀 Submit Question", interactive=True)
                    }

            app_functionality.change(
                fn=update_interface_based_on_mode,
                inputs=[app_functionality],
                outputs=[input_txt, text_submit_btn]
            )

# Custom CSS for better styling
demo.css = """
    #chatbot {
        border-radius: 10px;
        border: 1px solid #e0e0e0;
    }
    
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }
    
    .gr-button {
        border-radius: 8px;
        font-weight: 500;
    }
    
    .gr-textbox {
        border-radius: 8px;
    }
"""

if __name__ == "__main__":
    logger.info("Starting Gradio application")
    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True
    )
    logger.info("Gradio application stopped")