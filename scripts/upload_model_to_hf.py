# Upload the model to Hugging Face Hub

from huggingface_hub import HfApi, HfFolder

# Define path to model
model_path = "best_model.h5"

# Initialize Hugging Face API
api = HfApi()

token = "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"
HfFolder.save_token(token)

# Upload model to Hugging Face Hub
api.upload_folder(
    folder_path=model_path,
    path_in_repo="",
    repo_id="satriobagus/trashnet-model",
    repo_type="model"
)