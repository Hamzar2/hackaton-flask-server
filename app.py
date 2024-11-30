from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from huggingface_hub import InferenceClient
import json
import logging
import time
from transformers import pipeline
from PIL import Image

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
logging.basicConfig(level=logging.INFO)

client = InferenceClient(api_key="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")

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/chat', methods=['POST'])
def chat():
    try:
        # Validate the request
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400

        data = request.get_json()
        if 'messages' not in data or 'chatHistory' not in data:
            return jsonify({"error": "Missing 'messages' or 'chatHistory' in request body"}), 400

        # Extract chat history and current query
        chat_history = data['chatHistory']
        current_query = data['messages']

        logging.info("Received Current Query:")
        logging.info(json.dumps(current_query, indent=2))

        logging.info("Received Chat History:")
        logging.info(json.dumps(chat_history, indent=2))

        # Check if the current query contains classification results
        classification_message = None
        for msg in current_query:
            if 'Image Classification:' in msg['content']:
                classification_message = msg['content']
                break

        # Add classification-based system message if it exists
        if classification_message:
            # Begin constructing the medical report prompt
            medical_report_prompt = {
                "role": "assistant",
                "content": (
                    "You are a Medical AI Assistant tasked to help doctors make informed decisions and provide appropriate care, accurate medical reports, and recommendations. "
                    "Your response should follow standard medical reporting formats, providing clear and concise information. "
                    "Take into account the chat history. Here is the chat history for previous interactions:\n\n"
                )
            }

            # Add each conversation turn to the prompt
            # Ensure chat_history contains the expected format and handle empty or malformed entries
            for i, turn in enumerate(json.dumps(chat_history, indent=2)):
                if 'role' in turn and 'content' in turn:
                    # Add the formatted conversation turn to the prompt
                    medical_report_prompt["content"] += f"**Turn {i+1}:**\n**{turn['role'].capitalize()}:** {turn['content']}\n\n"
                else:
                    # Handle the case where the entry is malformed
                    logging.warning(f"Malformed conversation entry at turn {i+1}: {turn}")

            
            # Append classification results and the report generation instruction
            medical_report_prompt["content"] += (
                f"Ensure that the report is based on the classification results provided and is appropriate for medical professionals. "
                "If you encounter uncertainties, clearly state them and suggest next steps or further investigations. "
                "Include references at the end of the report as clickable links in markdown format (e.g., `[description](URL)`), ensuring they are sourced from credible platforms like PubMed, WHO, or CDC.\n\n"
                f"Image Classification Result: {classification_message}, model used: Devarshi/Brain_Tumor_Classification\n\n"
                "Please generate a medical report based on the result, including potential diagnoses, recommendations, and follow-up actions.\n\n"
                "Medical Report:"
            )
            current_query.append(medical_report_prompt)

        else:
            # Medical report prompt without classification results
            medical_report_prompt = {
                "role": "assistant",
                "content": (
                    "You are a Medical AI Assistant designed to help doctors make informed decisions, provide appropriate care, accurate medical reports, and recommendations. "
                    "Please provide guidance on how to proceed by suggesting the information or clarification needed to assist effectively. "
                    "Take into account the chat history. Here is the chat history for previous interactions:\n\n"
                )
            }

            # Add each conversation turn to the prompt
            # Ensure chat_history contains the expected format and handle empty or malformed entries
            for i, turn in enumerate(json.dumps(chat_history, indent=2)):
                if 'role' in turn and 'content' in turn:
                    # Add the formatted conversation turn to the prompt
                    medical_report_prompt["content"] += f"**Turn {i+1}:**\n**{turn['role'].capitalize()}:** {turn['content']}\n\n"
                else:
                    # Handle the case where the entry is malformed
                    logging.warning(f"Malformed conversation entry at turn {i+1}: {turn}")


            # Append general instruction for the assistant
            medical_report_prompt["content"] += (
                "Alternatively, describe general steps to analyze medical data if applicable. "
                "Use language suitable for medical professionals.\n\n"
                "Include references at the end of the report as clickable links in markdown format (e.g., `[description](URL)`), ensuring they are sourced from credible platforms like PubMed, WHO, or CDC.\n\n"
                "Instructions:"
            )
            current_query.append(medical_report_prompt)

        # Combine chat history and current query
        combined_prompt = current_query
        logging.info("Combined Prompt:")
        logging.info(json.dumps(combined_prompt, indent=2))

        # Generate response from the model
        def generate_response():
            stream = client.chat.completions.create(
                model="meta-llama/Meta-Llama-3-8B-Instruct",
                messages=combined_prompt,
                max_tokens=7090,
                stream=True
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        return Response(generate_response(), content_type='text/plain')

    except Exception as e:
        logging.error(f"Server error: {str(e)}")
        return jsonify({"error": str(e)}), 500



# Load image classification model
classifier = pipeline("image-classification", model="Devarshi/Brain_Tumor_Classification")

@app.route('/classify-image', methods=['POST'])
def classify_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']
    try:
        image = Image.open(image_file)
        results = classifier(image)
        label = results[0]['label']
        score = results[0]['score']
        return jsonify({"label": label, "score": score}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
