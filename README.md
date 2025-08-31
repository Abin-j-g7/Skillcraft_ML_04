🤖 Hand Gesture Recognition Model
📌 Overview

This project implements a Hand Gesture Recognition System that can accurately identify and classify different hand gestures from images. The model is built using Convolutional Neural Networks (CNNs) and trained on a dataset of hand gesture images.

The goal of this project is to enable intuitive human-computer interaction and gesture-based control systems.

📂 Dataset

Used the Hand Gesture Dataset provided by SkillCraft Technology
.

Dataset includes 20,000 images of hand gestures belonging to 10 different classes.

Images were split into:

14,000 for training

6,000 for validation & testing

🛠️ Tech Stack

Python 3.10+

TensorFlow / Keras – Deep Learning framework

NumPy, Pandas – Data handling

Matplotlib, Seaborn – Visualization

Scikit-learn – Evaluation metrics

🚀 Features

✔️ Preprocessed and augmented dataset for robust training
✔️ Built and trained CNN with multiple convolutional layers
✔️ Achieved ~99% test accuracy 🎯
✔️ Evaluated using confusion matrix & classification report
✔️ Supports real-time predictions on new images

📊 Results

Test Accuracy: 99.23%

Confusion Matrix: Showed strong class-wise performance

Classification Report: High precision, recall, and F1-score

▶️ Usage

Clone this repository:

git clone https://github.com/your-username/hand-gesture-recognition.git
cd hand-gesture-recognition


Install dependencies:

pip install -r requirements.txt


Train the model:

python train.py


Run predictions on sample images:

python predict.py --image path_to_image.jpg

📸 Demo

(Add some sample input-output images or GIFs here showing gesture recognition in action.)

🏆 Acknowledgments

SkillCraft Technology for providing the dataset and task.

TensorFlow & Keras community for open-source tools.

✨ This project is part of my AI/ML Internship with SkillCraft Technology.
