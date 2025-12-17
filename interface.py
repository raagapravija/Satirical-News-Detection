
"""
Satirical News Detection Interface

"""

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import anthropic
import os


MODEL_NAME = "claude-3-5-haiku-20241022"


model_root = "./sarcasm_model"

checkpoints = [d for d in os.listdir(model_root) if d.startswith("checkpoint-")]
latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
model_path = os.path.join(model_root, latest_checkpoint)

print(f"Loading model from: {model_path}")

# Tokenizer: load from base model 
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Model: load from checkpoint
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()
print("✓ Loaded trained model")

# Setup Claude client
api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    print("WARNING: ANTHROPIC_API_KEY not set. Claude predictions will fail.")
    client = None
else:
    client = anthropic.Anthropic(api_key=api_key)
    print("✓ Claude client initialized")

def predict_with_finetuned(text):
    """Predict using fine-tuned model"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pred = torch.argmax(probs, dim=-1).item()
    confidence = probs[0][pred].item()
    
    label = "Satirical" if pred == 1 else "Real News"
    return label, f"{confidence:.1%}"

def predict_with_claude_zero_shot(text):
    """Predict using Claude zero-shot (hidden from interface)"""
    if client is None:
        return "Error"
    
    prompt = f"""Is the following headline sarcastic or not sarcastic?

Headline: "{text}"

"""
    
    try:
        message = client.messages.create(
            model=MODEL_NAME,
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}]
        )
        response = message.content[0].text.strip().lower()
        
        if "sarcastic" in response and "not" not in response:
            return "Satirical"
        else:
            return "Real News"
    except Exception as e:
        return "Error"

def predict_all_methods(text):
    """Analyze text and return clean results"""
    if not text.strip():
        return "Please enter a headline", "", ""
    
    # Fine-tuned prediction
    ft_label, ft_conf = predict_with_finetuned(text)
    
    # Claude zero-shot (for internal comparison, not displayed)
    claude_label = predict_with_claude_zero_shot(text)
    
    # Text statistics
    word_count = len(text.split())
    
    # Build clean analysis
    analysis = f"""
**Classification**: {ft_label}

**Confidence**: {ft_conf}

**Text Length**: {word_count} words
"""
    
    return ft_label, ft_conf, analysis

# Custom CSS for professional styling
custom_css = """
#title {
    text-align: center;
    font-size: 3em;
    font-weight: 700;
    margin-top: 20px;
    margin-bottom: 10px;
    color: #1f2937;
}

#subtitle {
    text-align: center;
    font-size: 1.2em;
    color: #6b7280;
    margin-bottom: 40px;
}

.output-label {
    font-weight: 600;
    font-size: 1.1em;
}

.button-primary {
    font-size: 1.1em;
    padding: 12px 24px;
}
"""

# Create Gradio interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft(), title="Satirical News Detection") as demo:
    
    # Title
    gr.HTML('<h1 id="title">Satirical News Headline Detection</h1>')
    gr.HTML('<p id="subtitle">Distinguishing satirical content from real news using machine learning</p>')
    
    gr.Markdown("---")
    
    with gr.Row():
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="Enter News Headline",
                placeholder="Type or paste a news headline here...",
                lines=4,
                max_lines=6
            )
            
            submit_btn = gr.Button(
                "Analyze Headline", 
                variant="primary", 
                size="lg",
                elem_classes="button-primary"
            )
        
        with gr.Column(scale=1):
            gr.Markdown("### Analysis Results")
            
            prediction_output = gr.Textbox(
                label="Classification",
                lines=1,
                interactive=False,
                elem_classes="output-label"
            )
            
            confidence_output = gr.Textbox(
                label="Confidence Score",
                lines=1,
                interactive=False,
                elem_classes="output-label"
            )
            
            analysis_output = gr.Markdown(
                label="Details",
                value=""
            )
    
    submit_btn.click(
        fn=predict_all_methods,
        inputs=text_input,
        outputs=[prediction_output, confidence_output, analysis_output]
    )
    
    gr.Markdown("---")
    
    gr.Markdown("""
    <div style="text-align: center; color: #6b7280; font-size: 0.9em; margin-top: 30px;">
    <p><strong>Note:</strong> This system uses a fine-tuned DistilBERT model trained on labeled news headlines.</p>
    <p>Satirical sources include The Onion and similar publications. Real news includes major outlets like HuffPost and CNN.</p>
    </div>
    """)

# Launch the interface
if __name__ == "__main__":
    print("\n" + "="*60)
    print("LAUNCHING SATIRICAL NEWS DETECTION INTERFACE")
    print("="*60)
    
    print("="*60 + "\n")
    
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )