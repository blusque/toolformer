import os
from huggingface_hub import login

def login_to_huggingface():
    """
    Logs in to Hugging Face using the token stored in the environment variable
    `HF_TOKEN`. If the token is not set, it prompts the user to enter it.
    """
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        print("Logged in to Hugging Face using the token from environment variable.")
    else:
        hf_token = input("Enter your Hugging Face token: ")
        login(token=hf_token)
        print("Logged in to Hugging Face using the provided token.")

if __name__ == "__main__":
    login_to_huggingface()