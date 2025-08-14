# Evaluation pipelines for performance and bias
The two main files are _performance_eval.py_, which evaluates performace, and _bias_eval.py_, which evaluates bias (CrowS-Pairs and StereoSet), for any list of eligible Hugging Face LMs.

When running _bias_eval.py_, please install Language Model Evaluation Harness: 
```bash
pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git
```
When running _performance_eval.py_, please remember to run _performance_view.py_ afterwards to access the results.

Both evaluation-pipeline-2025 and StereoSet are folders adjusted to support our wrappers and modern Hugging Face models.  
