# Clone the repo
gh repo clone github.gatech.edu/mmathew32/supervised_learning_spring_22

# create env
conda env update --prefix ./env --file environment.yml  --prune

# activate env
conda activate path to env\env

# For Charts related to Credit Card Fraud Detection
python credit_card.py 


# For Charts related to Heart Rate Failure Detection
python heart_attack.py



# For Runtime Performance
python compare_runtime_models_CC_Fraud.py