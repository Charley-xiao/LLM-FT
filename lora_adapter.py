from mindnlp.transformers import AutoModel
from mindspore import nn, Parameter, ops, Tensor
from mindspore.train.serialization import save_checkpoint
import mindspore as ms

def load_llama3_model(model_name):
    model = AutoModel.from_pretrained(model_name)
    return model

class LoRAAdapter(nn.Cell):
    def __init__(self, model, r=4, alpha=32):
        super(LoRAAdapter, self).__init__()
        self.model = model
        self.r = r
        self.alpha = alpha
        self.lora_params = self.init_lora_params()

    def init_lora_params(self):
        lora_params = {}
        for name, param in self.model.parameters_and_names():
            if 'weight' in name:
                shape = param.shape
                lora_param = Parameter(Tensor(ops.Zeros()(shape, ms.float32)), name=name+'_lora')
                lora_params[name+'_lora'] = lora_param
        return lora_params

    def construct(self, *inputs):
        for name, param in self.lora_params.items():
            original_param = self.model.get_parameter(name.replace('_lora', ''))
            adjusted_param = original_param + param * (self.alpha / self.r)
            self.model.set_parameter(name.replace('_lora', ''), adjusted_param)
        return self.model(*inputs)
    
    def merge_lora_weights(self):
        for name, param in self.lora_params.items():
            original_param = self.model.get_parameter(name.replace('_lora', ''))
            self.model.set_parameter(name.replace('_lora', ''), original_param + param * (self.alpha / self.r))

    def save_model(self, save_path):
        save_checkpoint(self.model, save_path)
