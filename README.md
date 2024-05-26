# LLM-FT

```bash
# Create Python3.9 virtual environment
/home/ma-user/anaconda3/bin/conda create -n python-3.9.0 python=3.9.0 -y --override-channels --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
# Activate
source /home/ma-user/anaconda3/bin/activate python-3.9.0
# Clone LLM-FT repository
git clone https://github.com/Charley-xiao/LLM-FT.git
# Install requirements
cd LLM-FT
pip install -r requirements.txt
wget https://mindspore-demo.obs.cn-north-4.myhuaweicloud.com/mindnlp_install/mindnlp-0.3.1-py3-none-any.whl
pip install mindnlp-0.3.1-py3-none-any.whl
pip install tokenizers==0.15.2
# Get dataset (example: ruozhiba_gpt4, optional if able to connect to huggingface)
git clone https://huggingface.co/datasets/hfl/ruozhiba_gpt4
# Get model (optional if able to connect to huggingface)
git clone ...
# Fine-tune
export TOKENIZER_PARALLELISM=true
python finetune.py
```

dataset_sink_mode=True/False?

TOKENIZER_PARALLELISM=true/false?
