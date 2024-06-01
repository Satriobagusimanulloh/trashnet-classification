# Trash Classification with Deep Learning

This repository contains code for classifying trash types using deep learning with the TrashNet dataset. The model is developed in a Jupyter Notebook and includes steps for data preparation, exploratory image analysis, model training and evaluation, as well as model versioning and tracking using wandb.ai.

## Steps to Reproduce

### 1. Clone the Repository

```bash
git clone https://github.com/your_username/trash-classification.git
cd trash-classification
```

### 2. Set Up Virtual Environment and Install Dependencies

```bash
python -m venv venv
source venv/bin/activate  # For MacOS/Linux
.\venv\Scripts\activate  # For Windows
pip install -r requirements.txt
```

### 3. Download the Dataset

Download the TrashNet dataset from Hugging Face and place it in the data/ directory.

### 4. Run the Jupyter Notebook

```bash
python -m venv venv
source venv/bin/activate  # For MacOS/Linux
.\venv\Scripts\activate  # For Windows
pip install -r requirements.txt
```

### 5. GitHub Actions Workflow

The model development process is automated using GitHub Actions. Check the `.github/workflows/main.yml` file for the workflow details.

### 6. Publish the Model on Hugging Face Hub

After training the model, save it and upload it to the Hugging Face Hub using the provided script in the notebook.

### 7. Model Versioning and Tracking with wandb.ai

The notebook includes integration with wandb.ai for tracking experiments and model versions. Ensure you have a wandb account and API key set up.

## Additional Notes

- Ensure you have Python 3.11 installed on your system.
- For GPU support, install the appropriate version of TensorFlow and CUDA toolkit.
- Make sure to update the dataset path and wandb API key in the notebook before running.
- For any issues or questions, please open an issue or contact your_email@example.com.


Make sure to replace `your_username` with your GitHub username and provide the correct contact email in the contact section. This README.md will provide clear instructions for users to reproduce the trash classification model easily.