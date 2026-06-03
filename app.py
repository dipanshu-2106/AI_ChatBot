from flask import Flask, render_template, request, jsonify
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import logging

load_dotenv()

app = Flask(__name__)

# ================= LOGGING =================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================= API KEY =================
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    logger.error("GOOGLE_API_KEY not found!")
    raise ValueError("GOOGLE_API_KEY environment variable is required")

logger.info("✅ API Key loaded successfully")

# ================= LLM =================
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
    temperature=0.7
)

# ================= SIMPLE MEMORY =================
chat_history = []

# ================= ROUTES =================
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()

        if not user_message:
            return jsonify({'success': False, 'error': 'Message required'}), 400

        # Build context from history
        context = "You are a helpful AI assistant. Here is the conversation history:\n\n"
        for item in chat_history[-5:]:  # Last 5 messages
            context += f"User: {item['user']}\nAssistant: {item['bot']}\n\n"

        final_prompt = context + f"User: {user_message}\nAssistant:"

        # Get response
        response = llm.invoke(final_prompt)
        bot_reply = response.content

        # Save to memory
        chat_history.append({
            "user": user_message,
            "bot": bot_reply
        })

        logger.info(f"Chat: {user_message[:50]}...")

        return jsonify({
            'success': True,
            'message': bot_reply
        })

    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/clear', methods=['POST'])
def clear_chat():
    try:
        chat_history.clear()
        logger.info("Chat history cleared")
        return jsonify({'success': True, 'message': 'Chat cleared'})
    except Exception as e:
        logger.error(f"Error clearing chat: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/health')
def health():
    return jsonify({'status': 'healthy'}), 200


# ================= ERROR HANDLERS =================
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Server error'}), 500


# ================= RENDER ENTRY =================
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_ENV") == "development"
    
    logger.info(f"🚀 Starting server on port {port}")
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug_mode
    )