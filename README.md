# 🎨 Artistic Style Transfer

A web application that uses deep learning to transfer artistic styles from one image to another, creating stunning visual art pieces. Built with PyTorch and Flask, this project implements neural style transfer using the VGG19 architecture.

## 🚀 Live Demo

**[Try it live on Hugging Face Spaces](https://huggingface.co/spaces/praveen5001/artistic-style-transfer)**

## 🖼️ How It Works

The application uses a pre-trained VGG19 neural network to extract features from both content and style images. It then optimizes a target image to minimize:

1. **Content Loss**: Preserves the structure and objects from the content image
2. **Style Loss**: Applies the artistic style from the style image using Gram matrices

The optimization process runs for a specified number of iterations, balancing between content preservation and style application.

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Praveen9440/artistic-style-transfer.git
   cd artistic-style-transfer
   ```

2. **Create and activate conda environment**
   ```bash
   conda create -n style_transfer python=3.8
   conda activate style_transfer
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Open your browser**
   Navigate to `http://localhost:7860`

## 🎯 Usage

1. **Upload Images**: Select a content image (the subject) and a style image (the artistic style)
2. **Adjust Settings**:
   - **Iterations**: Higher values (300-500) = better quality but slower processing
   - **Style Weight**: Higher values = more artistic effect
3. **Click "Stylize Image"** and wait for the magic to happen!
4. **Download** your generated artwork

## 📋 Requirements

- `torch==2.0.1` - PyTorch deep learning framework
- `torchvision==0.15.2` - Computer vision utilities
- `numpy==1.24.3` - Numerical computing
- `Pillow==10.0.0` - Image processing
- `flask==2.3.3` - Web framework
- `werkzeug==2.3.7` - WSGI utilities
