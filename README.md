## Dataset Link: 
https://plg.uwaterloo.ca/~gvcormac/treccorpus07/

## Repo File Descriptions  
### Preprocessing  
__email_parsing.ipynb__ : Email parser  
__multiprocess_docs__ : Sped-up text processor  

### Transformers  
__dataset_prep__ : Prepping of data for bert/roberta training  
- Tokenisation
- Train-test Split
- Data Augmentation
- Undersampling
  
__transformers.ipynb__ : Bert/Roberta Finetuning and Results
- Weighted Eval Loss
- Weight Decay

## Docker File Commands  
__Building the Image__  
docker build -t transformers:latest .  
  
__Running the Container with GPU support__  
docker run -it --gpus all -h transformer_jax transformers:latest   
  
__Attaching to a running container in VSCode__  
https://code.visualstudio.com/docs/devcontainers/attach-container  
- Might need to have WSL2 installed

__Copying files from Container to local machine__  
Navigate to folder in local machine that you want to store the files in  
docker cp charming_newton:/app .  

