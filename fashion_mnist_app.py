"""
Fashion MNIST CNN Streamlit Application

This interactive web application demonstrates a Convolutional Neural Network (CNN)
for classifying Fashion MNIST images. Users can:
- Train a CNN model with configurable hyperparameters
- Visualize training progress and metrics in real-time
- Test the model with built-in or custom images
- Save and load trained models

Technical Implementation:
- Framework: PyTorch for deep learning, Streamlit for web interface
- Architecture: LeNet-style CNN with 2 conv blocks + 2 FC layers
- Dataset: Fashion MNIST (70,000 grayscale 28x28 images, 10 classes)
"""

import streamlit as st  # Web application framework for ML apps
import torch  # PyTorch deep learning framework
import torch.nn as nn  # Neural network modules
import torch.nn.functional as F  # Functional interface for layers
from torchvision import datasets, transforms  # Computer vision utilities
from torch.utils.data import DataLoader  # Data loading utilities
import numpy as np  # Numerical computing
import matplotlib.pyplot as plt  # Plotting and visualization
from PIL import Image  # Image processing library
import io  # For handling byte streams (model download)
import time  # For tracking training time


class FashionCNN(nn.Module):
    """
    Convolutional Neural Network for Fashion MNIST Classification

    Architecture Overview:
    - Input: 28×28×1 grayscale images
    - Conv Block 1: Conv(1→6, 5×5) → ReLU → MaxPool(2×2) → 12×12×6
    - Conv Block 2: Conv(6→16, 5×5) → ReLU → MaxPool(2×2) → 4×4×16
    - Classifier: Flatten(256) → FC(256→128) → ReLU → Dropout(0.2) → FC(128→10)

    Total Parameters: ~37K (lightweight model suitable for CPU training)
    """

    def __init__(self):
        super(FashionCNN, self).__init__()

        # Feature Extraction Layers
        # Conv1: Detects low-level features (edges, corners)
        # 1 input channel (grayscale), 6 output feature maps
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        # Pool1: Reduces spatial dimensions by 2x, preserves important features
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv2: Detects high-level features (shapes, patterns)
        # 6 input channels (from conv1), 16 output feature maps
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        # Pool2: Further dimension reduction
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Classification Layers
        # FC1: Maps flattened features (256) to hidden layer (128)
        self.fc1 = nn.Linear(16 * 4 * 4, 128)
        # FC2: Maps hidden layer to 10 output classes
        self.fc2 = nn.Linear(128, 10)
        # Dropout: Prevents overfitting by randomly dropping 20% of connections
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        """
        Forward pass through the network

        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)

        Returns:
            Logits tensor of shape (batch_size, 10)
        """
        # Feature extraction through conv blocks
        x = F.relu(self.conv1(x))  # Apply convolution and activation
        x = self.pool1(x)          # Downsample
        x = F.relu(self.conv2(x))  # Second conv block
        x = self.pool2(x)          # Final spatial reduction

        # Flatten for fully connected layers
        x = x.view(-1, 16 * 4 * 4)  # Reshape to (batch_size, 256)

        # Classification layers
        x = F.relu(self.fc1(x))     # Hidden layer with activation
        x = self.dropout(x)         # Apply dropout during training
        x = self.fc2(x)             # Output logits (no activation)

        return x


@st.cache_resource  # Cache data across app reruns for efficiency
def load_data():
    """
    Load and preprocess Fashion MNIST dataset

    The dataset contains:
    - 60,000 training images
    - 10,000 test images
    - 10 classes: T-shirt, Trouser, Pullover, Dress, Coat,
                  Sandal, Shirt, Sneaker, Bag, Ankle boot

    Preprocessing steps:
    1. Convert PIL images to tensors
    2. Normalize pixel values from [0,255] to [-1,1]
       This helps with gradient flow and training stability

    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    # Define preprocessing pipeline
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert PIL image to tensor, scales to [0,1]
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1,1] range
        # Formula: (pixel - mean) / std = (pixel - 0.5) / 0.5
    ])

    # Load training data (60,000 samples)
    train_dataset = datasets.FashionMNIST(
        root='./data',  # Directory to store downloaded data
        train=True,     # Load training split
        download=True,  # Download if not present
        transform=transform  # Apply preprocessing
    )

    # Load test data (10,000 samples)
    test_dataset = datasets.FashionMNIST(
        root='./data',
        train=False,    # Load test split
        download=True,
        transform=transform
    )

    return train_dataset, test_dataset


def train_model(model, train_loader, test_loader, epochs, device):
    """
    Train the CNN model with real-time progress visualization

    Training Process:
    1. Forward pass: Compute predictions
    2. Calculate loss using CrossEntropyLoss
    3. Backward pass: Compute gradients
    4. Update weights using Adam optimizer
    5. Evaluate on test set after each epoch

    Args:
        model: FashionCNN model instance
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        epochs: Number of training epochs
        device: CPU or GPU device for computation

    Returns:
        Tuple of (train_losses, train_accuracies, test_losses, test_accuracies)
    """
    # Loss function: CrossEntropyLoss combines LogSoftmax and NLLLoss
    # Perfect for multi-class classification
    criterion = nn.CrossEntropyLoss()

    # Optimizer: Adam adapts learning rate per parameter
    # lr=0.001 is a good default for Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Initialize metrics tracking lists
    train_losses = []      # Track loss per epoch
    train_accuracies = []  # Track accuracy per epoch
    test_losses = []       # Track test loss per epoch
    test_accuracies = []   # Track test accuracy per epoch

    # Streamlit UI elements for real-time updates
    progress_bar = st.progress(0)  # Visual progress indicator
    status_text = st.empty()       # Text showing current epoch
    metrics_placeholder = st.empty()  # Display training metrics

    # Main training loop
    for epoch in range(epochs):
        # Set model to training mode (enables dropout)
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        # Process training data in batches
        for batch_idx, (data, target) in enumerate(train_loader):
            # Move data to computation device (CPU/GPU)
            data, target = data.to(device), target.to(device)

            # Training step: forward + backward + optimize
            optimizer.zero_grad()  # Clear gradients from previous step
            output = model(data)   # Forward pass
            loss = criterion(output, target)  # Calculate loss
            loss.backward()        # Compute gradients
            optimizer.step()       # Update weights

            # Track training metrics
            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)  # Get predictions
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100. * correct / total
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)

        # Test evaluation phase
        # Set model to evaluation mode (disables dropout)
        model.eval()
        test_loss = 0
        correct = 0
        total = 0

        # Evaluate on test set without computing gradients (saves memory)
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)  # Forward pass only
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        avg_test_loss = test_loss / len(test_loader)
        test_accuracy = 100. * correct / total
        test_losses.append(avg_test_loss)
        test_accuracies.append(test_accuracy)

        # Update UI with training progress
        progress = (epoch + 1) / epochs
        progress_bar.progress(progress)  # Update progress bar
        status_text.text(f'Epoch {epoch + 1}/{epochs}')  # Show current epoch

        # Display real-time metrics
        metrics_placeholder.markdown(f"""
        **Epoch {epoch + 1} Metrics:**
        - Train Loss: {avg_train_loss:.4f}
        - Train Accuracy: {train_accuracy:.2f}%
        - Test Loss: {avg_test_loss:.4f}
        - Test Accuracy: {test_accuracy:.2f}%
        """)

    return train_losses, train_accuracies, test_losses, test_accuracies


def visualize_predictions(model, test_dataset, device, num_images=10):
    """
    Generate visualization of model predictions on random test images

    This function:
    1. Selects random images from the test set
    2. Makes predictions using the trained model
    3. Creates a grid visualization showing images with true/predicted labels

    Args:
        model: Trained CNN model
        test_dataset: Test dataset to sample from
        device: Computation device (CPU/GPU)
        num_images: Number of images to visualize

    Returns:
        matplotlib.figure.Figure: Grid of images with predictions
    """
    # Fashion MNIST class names for human-readable labels
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # Set model to evaluation mode (disables dropout)
    model.eval()

    # Select random images from test set
    indices = np.random.randint(0, len(test_dataset), num_images)

    # Create a 2x5 grid for displaying 10 images
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.flatten()

    # Make predictions without computing gradients (inference mode)
    with torch.no_grad():
        for idx, i in enumerate(indices):
            if idx >= 10:
                break

            # Get image and its true label
            image, label = test_dataset[i]
            # Add batch dimension and move to device
            image_input = image.unsqueeze(0).to(device)

            # Get model prediction
            output = model(image_input)
            _, predicted = torch.max(output, 1)  # Get class with highest score

            # Denormalize image for display (reverse the [-1, 1] normalization)
            img = image.squeeze().cpu().numpy()
            img = (img + 1) / 2  # Convert from [-1, 1] to [0, 1]

            # Display image with true and predicted labels
            axes[idx].imshow(img, cmap='gray')
            axes[idx].set_title(f'True: {class_names[label]}\nPred: {class_names[predicted.item()]}')
            axes[idx].axis('off')

    plt.tight_layout()
    return fig


def main():
    """
    Main Streamlit application function

    This function orchestrates the entire web application:
    1. Sets up the UI layout and configuration
    2. Handles data loading and preprocessing
    3. Manages model training and evaluation
    4. Provides visualization and testing capabilities
    5. Enables model saving/loading functionality
    """
    # Configure Streamlit page settings
    st.set_page_config(
        page_title="Fashion MNIST CNN",
        layout="wide",  # Use full screen width
        initial_sidebar_state="expanded"
    )

    # Application header
    st.title("🧠 Fashion MNIST CNN with PyTorch")
    st.markdown("""
    Train a Convolutional Neural Network to classify clothing items from the Fashion MNIST dataset.
    This interactive app demonstrates deep learning concepts in computer vision.
    """)

    # Sidebar configuration panel
    st.sidebar.header("⚙️ Model Configuration")

    # Hyperparameter controls
    # Batch size: Number of samples processed before weight update
    # Larger batch = more stable gradients but more memory
    batch_size = st.sidebar.slider(
        "Batch Size",
        min_value=32,
        max_value=256,
        value=100,
        step=32,
        help="Number of images to process in parallel. Higher values require more memory."
    )

    # Number of training epochs
    # Each epoch is one complete pass through the training data
    epochs = st.sidebar.slider(
        "Number of Epochs",
        min_value=1,
        max_value=20,
        value=10,
        help="Number of complete passes through the training dataset."
    )

    # Device selection (CPU vs GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    st.sidebar.info(f"💻 Using device: {device.type.upper()}")

    # Data Loading Section
    with st.spinner("📊 Loading Fashion MNIST dataset..."):
        # Load cached datasets (60K train + 10K test images)
        train_dataset, test_dataset = load_data()

    st.success(f"✅ Dataset loaded! Train: {len(train_dataset):,} samples, Test: {len(test_dataset):,} samples")

    # Create DataLoaders for batch processing
    # DataLoader handles batching, shuffling, and parallel loading
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Randomize order each epoch for better learning
        num_workers=0  # Use main process (set >0 for parallel loading)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Keep test order consistent for reproducibility
        num_workers=0
    )

    # Model Architecture Information
    with st.expander("🏗️ View Model Architecture"):
        st.markdown("### CNN Architecture Overview")
        st.code("""
        Input (28×28×1)
            ↓
        Conv2D(1→6, 5×5) → ReLU → MaxPool2D(2×2)  [Output: 12×12×6]
            ↓
        Conv2D(6→16, 5×5) → ReLU → MaxPool2D(2×2) [Output: 4×4×16]
            ↓
        Flatten() [Output: 256]
            ↓
        Linear(256→128) → ReLU → Dropout(0.2)
            ↓
        Linear(128→10) [Output: 10 class logits]
        """)

        # Calculate and display model complexity
        model = FashionCNN()
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Parameters", f"{total_params:,}")
        with col2:
            st.metric("Trainable Parameters", f"{trainable_params:,}")
        with col3:
            # Estimate model size in MB
            model_size_mb = (total_params * 4) / (1024 * 1024)  # 4 bytes per float32
            st.metric("Model Size", f"{model_size_mb:.2f} MB")

    # Training Section
    st.header("🎯 Model Training")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🚀 Start Training", type="primary", use_container_width=True):
            # Initialize new model instance
            model = FashionCNN().to(device)

            st.info("⏳ Training in progress... This may take a few minutes.")
            start_time = time.time()

            # Train model and collect metrics
            train_losses, train_accs, test_losses, test_accs = train_model(
                model, train_loader, test_loader, epochs, device
            )

            training_time = time.time() - start_time
            st.success(f"✅ Training completed in {training_time:.2f} seconds!")

            # Save model and metrics to session state for persistence
            # Session state maintains data across Streamlit reruns
            st.session_state['trained_model'] = model
            st.session_state['train_losses'] = train_losses
            st.session_state['train_accs'] = train_accs
            st.session_state['test_losses'] = test_losses
            st.session_state['test_accs'] = test_accs

            # Display final performance metrics
            col1_1, col1_2 = st.columns(2)
            with col1_1:
                st.metric(
                    "Final Training Accuracy",
                    f"{train_accs[-1]:.2f}%",
                    delta=f"+{train_accs[-1] - train_accs[0]:.2f}%"
                )
            with col1_2:
                st.metric(
                    "Final Test Accuracy",
                    f"{test_accs[-1]:.2f}%",
                    delta=f"+{test_accs[-1] - test_accs[0]:.2f}%"
                )

    with col2:
        if st.button("📂 Load Pre-trained Model", use_container_width=True):
            try:
                # Load saved model weights
                model = FashionCNN().to(device)
                model.load_state_dict(
                    torch.load('fashion_cnn_pytorch.pth', map_location=device)
                )
                st.session_state['trained_model'] = model
                st.success("✅ Pre-trained model loaded successfully!")
            except FileNotFoundError:
                st.error("❌ No pre-trained model found. Please train a model first.")
            except Exception as e:
                st.error(f"❌ Error loading model: {str(e)}")

    # Results Visualization Section
    # Only show if a model has been trained or loaded
    if 'trained_model' in st.session_state:
        st.header("📊 Training Results")

        if 'train_losses' in st.session_state:
            # Create visualizations of training metrics
            # These plots help identify overfitting and convergence
            col1, col2 = st.columns(2)

            with col1:
                # Loss curve visualization
                fig, ax = plt.subplots()
                epochs_range = range(1, len(st.session_state['train_losses']) + 1)
                ax.plot(epochs_range, st.session_state['train_losses'],
                       label='Train Loss', marker='o', markersize=4)
                ax.plot(epochs_range, st.session_state['test_losses'],
                       label='Test Loss', marker='s', markersize=4)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.set_title('Training and Test Loss')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

                # Interpretation helper
                if st.session_state['test_losses'][-1] > st.session_state['train_losses'][-1] * 1.2:
                    st.warning("⚠️ Model might be overfitting (test loss > train loss)")

            with col2:
                # Accuracy curve visualization
                fig, ax = plt.subplots()
                ax.plot(epochs_range, st.session_state['train_accs'],
                       label='Train Accuracy', marker='o', markersize=4)
                ax.plot(epochs_range, st.session_state['test_accs'],
                       label='Test Accuracy', marker='s', markersize=4)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Accuracy (%)')
                ax.set_title('Training and Test Accuracy')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

                # Performance interpretation
                final_test_acc = st.session_state['test_accs'][-1]
                if final_test_acc > 85:
                    st.success("🎉 Excellent model performance!")
                elif final_test_acc > 80:
                    st.info("👍 Good model performance")
                else:
                    st.warning("💡 Consider tuning hyperparameters")

        # Model Predictions Section
        st.header("🔍 Model Predictions")
        st.markdown("View how the model performs on random test images")

        if st.button("🎲 Generate Random Predictions", use_container_width=True):
            with st.spinner("Generating predictions..."):
                fig = visualize_predictions(
                    st.session_state['trained_model'],
                    test_dataset,
                    device
                )
                st.pyplot(fig)
                st.info("💡 Green titles = correct, Red titles = incorrect")

        # Model Saving Section
        st.header("💾 Save Model")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("💾 Save Model Locally", use_container_width=True):
                # Save model weights to local file
                torch.save(
                    st.session_state['trained_model'].state_dict(),
                    'fashion_cnn_pytorch.pth'
                )
                st.success("✅ Model saved as 'fashion_cnn_pytorch.pth'")

        with col2:
            # Prepare model for download
            buffer = io.BytesIO()
            torch.save(st.session_state['trained_model'].state_dict(), buffer)
            buffer.seek(0)

            # Download button for model file
            st.download_button(
                label="⬇️ Download Model File",
                data=buffer,
                file_name="fashion_cnn_pytorch.pth",
                mime="application/octet-stream",
                use_container_width=True,
                help="Download the trained model weights to your computer"
            )

    # Custom Image Testing Section
    st.header("🖼️ Test with Custom Image")
    st.markdown("""
    Upload your own image to test the model. The image will be automatically
    resized to 28×28 pixels and converted to grayscale.
    """)

    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg'],
        help="Works best with clothing items on plain backgrounds"
    )

    if uploaded_file is not None and 'trained_model' in st.session_state:
        # Process uploaded image
        image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
        image = image.resize((28, 28))  # Resize to model input size

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Uploaded Image (28×28)", width=200)
            st.info("📝 Image has been converted to grayscale and resized")

        with col2:
            # Prepare image for model inference
            # Apply same preprocessing as training data
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
            ])

            # Convert to tensor and add batch dimension
            img_tensor = transform(image).unsqueeze(0).to(device)

            # Get model and make prediction
            model = st.session_state['trained_model']
            model.eval()  # Set to evaluation mode

            # Inference without gradient computation
            with torch.no_grad():
                output = model(img_tensor)  # Get logits
                probabilities = F.softmax(output, dim=1)  # Convert to probabilities
                predicted_class = output.argmax(dim=1).item()  # Get predicted class

            # Class names for display
            class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                          'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

            # Display prediction results
            st.success(f"**Predicted Class:** {class_names[predicted_class]}")
            st.metric("Confidence", f"{probabilities[0][predicted_class]:.2%}")

            # Show probability distribution
            st.write("**Probability Distribution:**")
            # Create a bar chart of probabilities
            prob_data = {class_names[i]: probabilities[0][i].item()
                        for i in range(len(class_names))}
            sorted_probs = dict(sorted(prob_data.items(),
                                     key=lambda x: x[1], reverse=True))

            for class_name, prob in sorted_probs.items():
                # Show top 3 with emphasis
                if prob > 0.1:
                    st.write(f"• **{class_name}:** {prob:.2%}")
                else:
                    st.write(f"• {class_name}: {prob:.2%}")


# Application entry point
# This ensures the app only runs when executed directly, not when imported
if __name__ == "__main__":
    main()