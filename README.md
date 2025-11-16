# Autonomous-driving-mini-model
End-to-End Steering Angle Prediction Using Deep Learning

Overview

This project implements a compact, production-style end-to-end autonomous driving model that predicts a vehicle’s steering angle directly from front-camera images.
It is designed to be lightweight, modular, and demonstrates practical machine learning engineering skills suitable for real-world applications and portfolio use.

The system covers data preprocessing, augmentation, CNN-based regression, training loops, evaluation, checkpointing, and real-time inference on video.

⸻

Key Features
	•	PyTorch-based convolutional neural network for steering angle regression
	•	Data pipeline with augmentations (brightness, flips, resizing)
	•	CSV-based dataset structure for train/validation splits
	•	Evaluation using MAE and MSE
	•	Automatic checkpoint saving with best-model tracking
	•	Real-time inference on videos using OpenCV
	•	Steering angle overlay and visual indicator on output video

⸻

Project Structure

.
├── dataset.py          # Dataset loader and data augmentations
├── model.py            # CNN architecture for steering prediction
├── train.py            # Training loop and validation
├── inference.py        # Video inference script
├── utils.py            # Seed setting and checkpoint utilities
├── requirements.txt    # Dependencies
└── README.md


⸻

Dataset Format

You may use any self-driving dataset that includes front-facing camera images and corresponding steering angle labels.

Common options:
	•	Udacity Self-Driving Car Dataset
	•	comma.ai dataset
	•	Custom collected driving data

CSV Format

Each line should contain:

/path/to/image_001.jpg,0.034
/path/to/image_002.jpg,-0.120
/path/to/image_003.jpg,0.000

Steering angles may be in radians or degrees, but must be consistent throughout.

⸻

Installation

pip install -r requirements.txt


⸻

Training

Run the following command to train the model:

python train.py \
  --train_csv data/train.csv \
  --val_csv data/val.csv \
  --epochs 30 \
  --batch_size 64 \
  --lr 1e-4 \
  --save_dir checkpoints/

During training:
	•	The CNN learns to map images to steering values
	•	Augmentations are applied to improve generalization
	•	Validation MAE determines when the model is performing best
	•	The best-performing model is stored as best.pth

⸻

Inference on Video

To generate steering predictions on a driving video:

python inference.py \
  --model checkpoints/best.pth \
  --video input.mp4 \
  --out_video output_with_predictions.mp4

The output video contains the predicted steering angle and a line showing the steering direction.

⸻

Model Architecture

A compact CNN inspired by the NVIDIA end-to-end driving architecture is used.
It contains:
	•	Five convolutional layers
	•	ReLU activations
	•	Fully connected regression head

The architecture is designed for fast inference and suitability for embedded systems.

⸻

Evaluation Metrics

The model evaluates performance using:
	•	Mean Squared Error (training loss)
	•	Mean Absolute Error (validation metric)

MAE provides a clearer interpretation of steering accuracy.

⸻

Customization

You can modify:
	•	Input resolution
	•	Network depth
	•	Regularization and dropout
	•	Data augmentations
	•	Steering normalization

Possible extensions include:
	•	Multi-camera training
	•	Temporal models (CNN + LSTM)
	•	Exporting to ONNX or TFLite
	•	Advanced augmentation (shadows, cropping, Gaussian blur)

⸻
