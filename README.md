# 🎨 Artistic Style Transfer

A web application that uses deep learning to transfer artistic styles from one image to another, creating stunning visual art pieces. Built with PyTorch and Flask, this project implements neural style transfer using the VGG19 architecture.

## 🌟 Features

- **Real-time Style Transfer**: Upload any content and style images to create artistic combinations
- **Adjustable Parameters**: Fine-tune the style transfer with customizable iterations and style weight
- **Modern Web Interface**: Beautiful, responsive UI with drag-and-drop functionality
- **Example Gallery**: Try pre-loaded examples to see the capabilities
- **Download Results**: Save your generated artwork directly
- **GPU Acceleration**: Automatically uses CUDA if available for faster processing
- **Mobile Responsive**: Works seamlessly on desktop and mobile devices

## 🚀 Live Demo

**[Try it live on Hugging Face Spaces](https://huggingface.co/spaces/praveen5001/artistic-style-transfer)**

## 🖼️ How It Works

The application uses a pre-trained VGG19 neural network to extract features from both content and style images. It then optimizes a target image to minimize:

1. **Content Loss**: Preserves the structure and objects from the content image
2. **Style Loss**: Applies the artistic style from the style image using Gram matrices

The optimization process runs for a specified number of iterations, balancing between content preservation and style application.

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for faster processing)
- Conda or pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/artistic-style-transfer.git
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

## 📋 Requirements

- `torch==2.0.1` - PyTorch deep learning framework
- `torchvision==0.15.2` - Computer vision utilities
- `numpy==1.24.3` - Numerical computing
- `Pillow==10.0.0` - Image processing
- `flask==2.3.3` - Web framework
- `werkzeug==2.3.7` - WSGI utilities

## 🎯 Usage

### Basic Usage

1. **Upload Images**: Select a content image (the subject) and a style image (the artistic style)
2. **Adjust Settings**:
   - **Iterations**: Higher values (300-500) = better quality but slower processing
   - **Style Weight**: Higher values = more artistic effect
3. **Click "Stylize Image"** and wait for the magic to happen!
4. **Download** your generated artwork

### Tips for Best Results

- **Content Images**: Use clear, well-lit photos with distinct objects
- **Style Images**: Choose images with strong artistic patterns, textures, or brushstrokes
- **Resolution**: Images are automatically resized to 512px max for optimal performance
- **Processing Time**: Expect 1-3 minutes depending on your hardware and settings

## 🏗️ Project Structure

```
artistic-style-transfer/
├── app.py                 # Flask web application
├── style_transfer_model.py # Neural style transfer implementation
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html        # Web interface
├── static/
│   ├── style.css         # Styling
│   ├── example1_*.jpg    # Example images
│   └── examples/         # Additional example images
└── README.md             # This file
```

## 🔧 Technical Details

### Architecture

- **Backend**: Flask web server with RESTful API
- **AI Model**: VGG19 feature extractor with custom optimization
- **Frontend**: Vanilla JavaScript with modern CSS
- **Processing**: PyTorch with optional CUDA acceleration

### Key Components

- **VGGFeatureExtractor**: Extracts features from multiple VGG19 layers
- **Content Loss**: MSE between content and target features
- **Style Loss**: MSE between Gram matrices of style and target features
- **Optimization**: Adam optimizer with configurable parameters

### Performance

- **CPU**: ~3-5 minutes for 300 iterations
- **GPU**: ~30-60 seconds for 300 iterations
- **Memory**: ~2-4GB RAM usage
- **Image Size**: Automatically resized to 512px max dimension

## 🚀 Deployment

### Hugging Face Spaces

This project is optimized for deployment on Hugging Face Spaces:

1. Create a new Space on Hugging Face
2. Upload all project files
3. Set the Space to use Python SDK
4. The app will automatically run on port 7860

### Local Deployment

For production deployment:

```bash
# Set environment variables
export FLASK_ENV=production
export PORT=7860

# Run with gunicorn (recommended)
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:7860 app:app
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PyTorch Team** for the excellent deep learning framework
- **VGG Team** for the pre-trained VGG19 model
- **Flask Team** for the lightweight web framework
- **Hugging Face** for providing free hosting for ML applications

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/your-username/artistic-style-transfer/issues) page
2. Create a new issue with detailed information
3. For urgent matters, contact [your-email@example.com]

## 🔮 Future Enhancements

- [ ] Multiple style transfer algorithms
- [ ] Batch processing capabilities
- [ ] Video style transfer
- [ ] Advanced parameter controls
- [ ] User authentication and galleries
- [ ] API endpoints for integration

---

**Made with ❤️ and PyTorch**

*Transform your photos into artistic masterpieces!*
