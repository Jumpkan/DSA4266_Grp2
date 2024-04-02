## Docker File Commands  
__Building the Image__  
docker build -t transformers:latest .  
__Running the Container with GPU support__  
docker run -it --gpus all transformers:latest -h transformer_jax  
__Attaching to a running container in VSCode__  
https://code.visualstudio.com/docs/devcontainers/attach-container  
- Might need to have WSL2 installed

__Copying files from Container to local machine__  
Navigate to folder in local machine that you want to store the files in  
docker cp charming_newton:/app .  
