"""
CycleGAN Face ↔ Sketch Converter Flask Application

This Flask app provides a user-friendly interface for converting between
real face images and sketches using a trained CycleGAN model.

Features:
- Upload images via file input
- Live camera input support
- Automatic detection of sketch vs real face
- Real-time image conversion
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import os
import base64
from io import BytesIO

# ========================================
# Flask App Configuration
# ========================================
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['SECRET_KEY'] = 'cyclegan-secret-key'

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('./static/results', exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ========================================
# Model Architecture Definitions
# ========================================
class ResidualBlock(nn.Module):
    """Residual block for the generator"""
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels)
        )
    
    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    """Generator network for CycleGAN (ResNet-based)"""
    def __init__(self, input_channels=3, n_residual_blocks=9):
        super(Generator, self).__init__()
        
        # Initial convolution
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]
        
        # Downsampling
        in_channels = 64
        for _ in range(2):
            out_channels = in_channels * 2
            model += [
                nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]
            in_channels = out_channels
        
        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_channels)]
        
        # Upsampling
        for _ in range(2):
            out_channels = in_channels // 2
            model += [
                nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]
            in_channels = out_channels
        
        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, input_channels, 7),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)


# ========================================
# Model Loading
# ========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_SIZE = 256

# Initialize models
G_AB = Generator().to(device)  # Face -> Sketch
G_BA = Generator().to(device)  # Sketch -> Face

def load_models():
    """Load trained models from saved checkpoints"""
    global G_AB, G_BA
    
    # Try to load from individual model files first
    g_ab_path = './G_AB_final.pth'
    g_ba_path = './G_BA_final.pth'
    checkpoint_path = './final_model_checkpoint.pth'
    
    try:
        # Try loading from individual files
        if os.path.exists(g_ab_path) and os.path.exists(g_ba_path):
            print("Loading models from individual files...")
            G_AB.load_state_dict(torch.load(g_ab_path, map_location=device))
            G_BA.load_state_dict(torch.load(g_ba_path, map_location=device))
            print("✓ Models loaded from individual files")
        # Try loading from combined checkpoint
        elif os.path.exists(checkpoint_path):
            print("Loading models from combined checkpoint...")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            G_AB.load_state_dict(checkpoint['G_AB_state_dict'])
            G_BA.load_state_dict(checkpoint['G_BA_state_dict'])
            print("✓ Models loaded from checkpoint")
        else:
            print("⚠ Warning: No model files found!")
            print(f"  Looking for: {g_ab_path}, {g_ba_path}, or {checkpoint_path}")
            return False
        
        # Set models to evaluation mode
        G_AB.eval()
        G_BA.eval()
        print(f"✓ Models loaded successfully on {device}")
        return True
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        import traceback
        traceback.print_exc()
        return False

# Load models at startup
print("="*60)
print("Initializing CycleGAN Face ↔ Sketch Converter")
print("="*60)
print(f"Device: {device}")
if not load_models():
    print("⚠ Warning: Models not loaded. Some features may not work.")


# ========================================
# Image Preprocessing and Postprocessing
# ========================================
inference_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def tensor_to_image(tensor):
    """Convert normalized tensor to PIL Image"""
    tensor = tensor.squeeze(0).cpu()
    tensor = (tensor + 1) / 2.0  # Denormalize from [-1, 1] to [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    to_pil = transforms.ToPILImage()
    return to_pil(tensor)


def detect_image_type(image):
    """
    Automatically detect if image is a sketch or real face.
    
    Uses variance of pixel values - sketches typically have lower variance
    due to limited color palette and simpler structure.
    
    Args:
        image: PIL Image
        
    Returns:
        'sketch' or 'face'
    """
    # Convert to grayscale for analysis
    img_array = np.array(image.convert('L'))
    
    # Calculate variance (sketches usually have lower variance)
    variance = np.var(img_array)
    
    # Calculate edge density (sketches usually have higher edge density)
    # Simple edge detection using gradient
    grad_x = np.abs(np.gradient(img_array.astype(float), axis=1))
    grad_y = np.abs(np.gradient(img_array.astype(float), axis=0))
    edge_density = np.mean(grad_x + grad_y)
    
    # Combined heuristic
    # Sketches: low variance, high edge density
    # Faces: higher variance (more details), moderate edge density
    threshold_variance = 800
    threshold_edge = 15
    
    is_sketch = (variance < threshold_variance) or (edge_density > threshold_edge)
    
    return 'sketch' if is_sketch else 'face'


def convert_image(image, model_type='auto'):
    """
    Convert image from one domain to another.
    
    Args:
        image: PIL Image
        model_type: 'auto' (detect), 'face2sketch', or 'sketch2face'
    
    Returns:
        tuple: (converted_image_display, converted_image_download, model_type_used, detected_type)
               - converted_image_display: resized to original dimensions (for display)
               - converted_image_download: at 256x256 (original model output, for download)
    """
    # Store original dimensions
    original_size = image.size  # (width, height)
    
    # Detect image type if auto
    detected_type = None
    if model_type == 'auto':
        detected_type = detect_image_type(image)
        model_type = 'face2sketch' if detected_type == 'face' else 'sketch2face'
        print(f"Detected image type: {detected_type}, using model: {model_type}")
    
    # Preprocess (resizes to IMAGE_SIZE x IMAGE_SIZE)
    image_tensor = inference_transform(image).unsqueeze(0).to(device)
    
    # Convert
    with torch.no_grad():
        if model_type == 'face2sketch':
            converted_tensor = G_AB(image_tensor)
        else:  # sketch2face
            converted_tensor = G_BA(image_tensor)
    
    # Convert back to PIL Image (256x256 - original model output)
    converted_image_download = tensor_to_image(converted_tensor)
    
    # Resize for display (resize back to original dimensions)
    converted_image_display = converted_image_download.resize(original_size, Image.LANCZOS)
    
    return converted_image_display, converted_image_download, model_type, detected_type


# ========================================
# Flask Routes
# ========================================
@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and conversion"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Read image
            image = Image.open(file.stream).convert('RGB')
            
            # Convert (returns display version and download version)
            converted_image_display, converted_image_download, model_type, detected_type = convert_image(image)
            
            # Save images
            input_filename = secure_filename(file.filename)
            output_filename = 'converted_' + input_filename
            
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
            output_path = os.path.join('./static/results', output_filename)
            
            image.save(input_path)
            # Save the 256x256 version (for download)
            converted_image_download.save(output_path)
            
            # Convert display version to base64 for response (resized to original)
            buffered = BytesIO()
            converted_image_display.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            return jsonify({
                'success': True,
                'image': 'data:image/png;base64,' + img_str,
                'model_used': model_type,
                'detected_type': detected_type,
                'input_path': f'/uploads/{input_filename}',
                'output_path': f'/results/{output_filename}'  # This will serve 256x256 version for download
            })
            
        except Exception as e:
            print(f"Error processing image: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Processing error: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/camera', methods=['POST'])
def process_camera():
    """Handle camera input and conversion"""
    data = request.json
    if 'image' not in data:
        return jsonify({'error': 'No image data provided'}), 400
    
    try:
        # Decode base64 image
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        
        # Convert (returns display version and download version)
        converted_image_display, converted_image_download, model_type, detected_type = convert_image(image)
        
        # Convert display version to base64 (resized to original for display)
        buffered = BytesIO()
        converted_image_display.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Also provide 256x256 version for download if needed
        buffered_download = BytesIO()
        converted_image_download.save(buffered_download, format="PNG")
        img_str_download = base64.b64encode(buffered_download.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'image': 'data:image/png;base64,' + img_str,  # Display version (resized)
            'image_download': 'data:image/png;base64,' + img_str_download,  # 256x256 for download
            'model_used': model_type,
            'detected_type': detected_type
        })
        
    except Exception as e:
        print(f"Error processing camera image: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/results/<filename>')
def result_file(filename):
    """Serve result files"""
    return send_from_directory('./static/results', filename)


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'device': str(device),
        'models_loaded': G_AB is not None and G_BA is not None
    })


# ========================================
# Main Entry Point
# ========================================
if __name__ == '__main__':
    print("\n" + "="*60)
    print("Flask App Starting...")
    print("="*60)
    print("Access the application at: http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)

