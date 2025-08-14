# Scripts for pre-training and uploading LTG models on HPC
Python files:
- _run_mlm.py:_ Standard MLM pre-training code
- _upload.py:_ Code for uploadign the pre-trained model onto Hugging Face

Slurm files:
- _slurm_train_: allows to define the properties (and GPUs used) of a training run
- _slurm_upload_: allows to define the name and the repo of the uploaded model
