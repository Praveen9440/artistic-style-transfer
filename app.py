import os
import time
import json
import shutil
import tempfile
from flask import Flask, request, render_template, jsonify, send_from_directory, session
from werkzeug.utils import secure_filename
from style_transfer_model import run_style_transfer

app = Flask(__name__)
# A secret key is required to use Flask's session feature
app.config['SECRET_KEY'] = 'change-this-to-a-random-secret-string'

# Use temporary directories for Hugging Face Spaces
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp(prefix='uploads_')
app.config['OUTPUT_FOLDER'] = tempfile.mkdtemp(prefix='outputs_')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# No need for example storage - we only have one static example

def cleanup_user_files():
    """Clean up files from uploads and outputs folders."""
    try:
        # Clean uploads folder
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        
        # Clean outputs folder
        for filename in os.listdir(app.config['OUTPUT_FOLDER']):
            file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
    except Exception as e:
        print(f"Error cleaning up user files: {e}")

# Single static example - no initialization needed

@app.route('/')
def index():
    """Renders the main HTML page."""
    return render_template('index.html')

@app.route('/stylize', methods=['POST'])
def stylize():
    """Handles the image upload, runs the model, and cleans up old files."""
    # --- File Cleanup Logic ---
    # Clean up all files from uploads and outputs folders
    cleanup_user_files()
    
    # Check if there are old files from this user's previous session and delete them
    if 'previous_files' in session:
        for file_path in session['previous_files']:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error deleting old file {file_path}: {e}")
        session.pop('previous_files') # Clear the list from the session

    # --- File Validation Logic ---
    if 'content_image' not in request.files or 'style_image' not in request.files:
        return jsonify({'error': 'Missing image files'}), 400

    content_file = request.files['content_image']
    style_file = request.files['style_image']

    if content_file.filename == '' or style_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # --- Get Settings and Prepare Files ---
    num_steps = request.form.get('num_steps', default=300, type=int)
    style_weight = request.form.get('style_weight', default=1000000, type=int)

    timestamp = int(time.time())
    content_filename = secure_filename(f"content_{timestamp}.jpg")
    style_filename = secure_filename(f"style_{timestamp}.jpg")
    output_filename = secure_filename(f"output_{timestamp}.jpg")

    content_path = os.path.join(app.config['UPLOAD_FOLDER'], content_filename)
    style_path = os.path.join(app.config['UPLOAD_FOLDER'], style_filename)
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

    content_file.save(content_path)
    style_file.save(style_path)

    # --- Run the AI Model ---
    print(f"Starting stylization with {num_steps} steps and style weight {style_weight}")
    run_style_transfer(
        content_img_path=content_path,
        style_img_path=style_path,
        output_img_path=output_path,
        num_steps=num_steps,
        style_weight=style_weight,
        content_weight=1
    )

    # --- Store file paths in session for next cleanup ---
    session['previous_files'] = [content_path, style_path, output_path]

    return jsonify({'output_url': f"/{app.config['OUTPUT_FOLDER']}{output_filename}"})

@app.route('/outputs/<filename>')
def send_output_file(filename):
    """Serves the generated output image."""
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serves static files."""
    return send_from_directory('static', filename)

# Removed post-example functionality - no need to store examples

@app.route('/get-examples')
def get_examples():
    """Returns the single static example."""
    try:
        # Return the single static example
        static_example = {
            "id": "static_1",
            "content": "/static/example1_content.jpeg",
            "style": "/static/example1_style.jpg",
            "output": "/static/example1_output.jpg",
            "is_user_generated": False,
            "created_at": 0
        }
        
        return jsonify({
            'example': static_example,
            'has_more': False,
            'total_examples': 1
        })
        
    except Exception as e:
        print(f"Error getting example: {e}")
        return jsonify({'error': 'Failed to load example'}), 500

@app.route('/load-example', methods=['POST'])
def load_example():
    """Load example images and result directly."""
    try:
        # Define the static example paths
        content_path = 'static/example1_content.jpeg'
        style_path = 'static/example1_style.jpg'
        output_path = 'static/example1_output.jpg'
        
        # Check if files exist
        if not all(os.path.exists(path) for path in [content_path, style_path, output_path]):
            return jsonify({'error': 'Example files not found'}), 404
        
        return jsonify({
            'success': True,
            'content_url': '/static/example1_content.jpeg',
            'style_url': '/static/example1_style.jpg',
            'output_url': '/static/example1_output.jpg',
            'message': 'Example loaded successfully!'
        })
        
    except Exception as e:
        print(f"Error loading example: {e}")
        return jsonify({'error': 'Failed to load example'}), 500

@app.route('/cleanup', methods=['POST'])
def cleanup():
    """Clean up user files when they leave the page."""
    try:
        cleanup_user_files()
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error during cleanup: {e}")
        return jsonify({'error': 'Cleanup failed'}), 500

if __name__ == '__main__':
    # For Hugging Face Spaces, use port 7860
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port, debug=False)