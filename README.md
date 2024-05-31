# 基于MindSpore的问答式大模型LoRA微调

这个项目是一个基于MindSpore的问答式大模型微调的工具，使用LoRA算法。它可以帮助你快速地微调你的模型，以便在你的特定任务上获得更好的性能。

## 快速上手

将微调的数据集以及模型放在项目根目录下，修改 `finetune.py` 中的参数，执行下面的指令：

```bash
# 创建 Python 3.9 环境
/home/ma-user/anaconda3/bin/conda create -n python-3.9.0 python=3.9.0 -y --override-channels --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
source /home/ma-user/anaconda3/bin/activate python-3.9.0
# 克隆本仓库
git clone https://github.com/Charley-xiao/LLM-FT.git
# 安装所需库
cd LLM-FT
pip install -r requirements.txt
wget https://mindspore-demo.obs.cn-north-4.myhuaweicloud.com/mindnlp_install/mindnlp-0.3.1-py3-none-any.whl
pip install mindnlp-0.3.1-py3-none-any.whl
pip install tokenizers==0.15.2
# 获取数据集（示例：弱智吧）
git clone https://huggingface.co/datasets/hfl/ruozhiba_gpt4
# 获取模型
git clone ...
# 开始微调
export TOKENIZER_PARALLELISM=true
python finetune.py
```

## 如何贡献

我们欢迎任何形式的贡献，包括但不限于提交问题、提供解决方案、改进代码和文档。

## 许可证

这个项目是在MIT许可证下发布的。详情请参考LICENSE文件。
