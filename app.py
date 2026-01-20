import gradio as gr
from answer import stream_answer_question

# 1. Define Knowledge Base of Reference Answers
REF_ANSWERS = {
    "What imaging method was used to track glomerular dynamics and cell movement in vivo in podocyte research?": "Serial multiphoton microscopy (MPM) was used to visualize podocytes and PEC migration in vivo.",
    "Which two proteins coordinate ER-endosome contacts to regulate PI4P and affect retromer-/WASH-dependent budding?": "VAP (VAPA/VAPB) coordinates endosome contacts with OSBP to regulate PI4P, which controls retromer-/WASH-dependent budding from endosomes.",
    "What effect did leptin treatment have on body weight in ob/ob mice after 5 days?": "Leptin treatment reduced body weight by 13.2% after 5 days in ob/ob mice.",
}


# 2. Formatting Helpers
def format_display(context, reference_answer=None):
    """Formats both the Reference Answer (if valid) and the Retrieved Context for the side panel."""
    html_content = ""

    # Section 1: Reference Answer (Ground Truth)
    if reference_answer:
        html_content += (
            "<div style='margin-bottom: 20px; padding: 15px; background-color: #e6f3ff; border-left: 5px solid #2b6cb0; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);'>"
            "<h3 style='color: #2b6cb0; margin-top: 0; display: flex; align-items: center;'>📝 Reference Answer <span style='font-size:0.8em; font-weight:normal; margin-left:8px; color:#555;'>(Ground Truth)</span></h3>"
            f"<p style='font-size: 1.05em; line-height: 1.5; margin-bottom: 0; color: #2d3748;'>{reference_answer}</p>"
            "</div>"
        )

    # Section 2: Retrieved Context
    html_content += "<h3 style='color: #ed8936; border-bottom: 2px solid #ed8936; padding-bottom: 8px;'>📄 Retrieved Evidence</h3>"

    if not context:
        html_content += "<p style='color: #718096; font-style: italic;'>Searching for relevant documents...</p>"
        return html_content

    for i, doc in enumerate(context):
        # Handle cases where doc might be a dict or a Document object
        content = getattr(doc, "page_content", str(doc))
        id = doc.id
        meta = getattr(doc, "metadata", {})
        doc_id = meta.get("id", i + 1)  # Fallback to index if ID missing
        title = meta.get("title", "Untitled Document")

        html_content += (
            "<div style='margin-bottom: 12px; padding: 12px; border: 1px solid #e2e8f0; border-radius: 6px; background-color: #fff;'>"
            f"<div style='font-size: 0.85em; color: #ed8936; font-weight: bold; margin-bottom: 4px;'>SOURCE {i+1}</div>"
            f"<div style='font-size: 0.85em; color: #718096; margin-bottom: 4px;'>ID: {id}</div>"
            f"<div style='font-size: 0.95em; font-weight: 600; color: #2d3748; margin-bottom: 6px;'>{title}</div>"
            f"<div style='font-size: 0.9em; color: #4a5568; line-height: 1.4;'>{content[:300]}...</div>"
            "</div>"
        )

    return html_content


def process_query(message, history):
    """
    Unified handler that manages history updates and streaming response.
    """

    if history is None:
        history = []
    if not message:
        yield history, ""
        return

    # 1. Update History with User Message
    history.append({"role": "user", "content": message})

    # Check for Reference Answer
    ref_answer = REF_ANSWERS.get(message.strip())

    # Yield immediately so user sees their question
    yield history, format_display(None, ref_answer)

    # 2. Prepare Assistant Response
    history.append({"role": "assistant", "content": ""})
    full_response = ""
    docs = []

    # 3. Stream from Backend
    # passing history[:-2] (prior history) and message
    prior_history = history[:-2]

    for chunk, retrieved_docs in stream_answer_question(message, prior_history):
        full_response += chunk

        # Capture docs when they become available
        if retrieved_docs:
            docs = retrieved_docs

        # Update UI
        history[-1]["content"] = full_response

        # Update Info Panel with accumulated docs
        yield history, format_display(docs, ref_answer)


def main():
    theme = gr.themes.Soft(
        primary_hue="blue",
        neutral_hue="slate",
        font=["Inter", "system-ui", "sans-serif"],
    )

    with gr.Blocks(
        title="SciFacts Assistant",
        theme=theme,
        css=".contain { max-width: 95% !important; }",
    ) as ui:

        # --- Header ---
        with gr.Row():
            with gr.Column():
                gr.Markdown(
                    """
                    # 🧬 SciFacts Expert Assistant
                    ### Verify scientific claims with high-precision RAG
                    """
                )

        # --- System Architecture Info ---
        with gr.Accordion("ℹ️ System Architecture & Model Details", open=False):
            with gr.Row():
                with gr.Column():
                    gr.Markdown(
                        """
                        **🤖 Main Chat Model: Kimi-2**
                        * **Model:** `moonshotai/kimi-k2-instruct-0905`
                        * **Why:** State-of-the-art long-context understanding.
                        * [🔗 Official Kimi Documentation](https://moonshotai.github.io/Kimi-K2/)
                        """
                    )
                with gr.Column():
                    gr.Markdown(
                        """
                        **⚖️ Reranker: GPT-OSS-120b**
                        * **Model:** `openai/gpt-oss-120b` (via Groq)
                        * **Function:** Re-scores retrieved documents for relevance.
                        """
                    )
                with gr.Column():
                    gr.Markdown(
                        """
                        **📚 Dataset: SciFacts**
                        * **Source:** Articles which answer scientific claims.
                        * **Embeddings:** `all-MiniLM-L6-v2`
                        * [🔗 Visit SciFacts Project](https://ir-datasets.com/beir.html#beir/scifact)
                        """
                    )

        # --- FIX: Define components BEFORE layout to handle dependencies ---
        # We create info_panel here with render=False so we can reference it
        # in the Left Column (for Examples), but display it in the Right Column later.

        info_panel = gr.HTML(
            label="Retrieved Context", value=format_display([]), render=False
        )

        chatbot = gr.Chatbot(
            label="Conversation",
            height=600,
            type="messages",
            show_copy_button=True,
            avatar_images=(
                None,
                "https://em-content.zobj.net/source/emojidex/107/female-scientist-type-4_1f469-1f3fd-200d-1f52c.png",
            ),
            render_markdown=True,
            render=False,
        )

        txt_input = gr.Textbox(
            scale=4,
            show_label=False,
            placeholder="Type your scientific question here...",
            container=False,
            render=False,
        )

        btn_submit = gr.Button("Submit", scale=1, variant="primary", render=False)

        # --- Layout ---
        with gr.Row():
            # LEFT: Chat Interface
            with gr.Column(scale=3):
                chatbot.render()  # Display chatbot

                with gr.Row():
                    txt_input.render()  # Display input
                    btn_submit.render()  # Display button

                # Reference Questions
                gr.Markdown("### 🔍 Try a Reference Question:")

                example_questions = [[q] for q in REF_ANSWERS.keys()]

                # Now this works because info_panel is already defined (in memory)
                gr.Examples(
                    examples=example_questions,
                    inputs=txt_input,
                    outputs=[chatbot, info_panel],
                    fn=process_query,
                    run_on_click=True,
                    cache_examples=False,
                )

            # RIGHT: Info Panel
            with gr.Column(scale=2):
                info_panel.render()  # Display info_panel

        # --- Event Wiring ---
        txt_input.submit(
            process_query, inputs=[txt_input, chatbot], outputs=[chatbot, info_panel]
        ).then(lambda: "", None, txt_input)

        btn_submit.click(
            process_query, inputs=[txt_input, chatbot], outputs=[chatbot, info_panel]
        ).then(lambda: "", None, txt_input)

    ui.launch(inbrowser=True)


if __name__ == "__main__":
    main()
