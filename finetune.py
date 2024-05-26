import mindspore as ms
from mindspore.train import Model
from mindspore.train.callback import LossMonitor
from mindspore.nn import SoftmaxCrossEntropyWithLogits, Adam

from make_dataset import load_and_preprocess_dataset
from lora_adapter import load_llama3_model, LoRAAdapter

def finetune_llama3_with_lora(dataset_name, model_name, tokenizer_name, save_path, epochs=3, batch_size=8, learning_rate=1e-4):
    dataset = load_and_preprocess_dataset(dataset_name, tokenizer_name)
    
    model = load_llama3_model(model_name)
    lora_model = LoRAAdapter(model)

    loss_fn = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    optimizer = Adam(lora_model.trainable_params(), learning_rate=learning_rate)

    train_model = Model(lora_model, loss_fn, optimizer)

    train_model.train(epochs, dataset, callbacks=[LossMonitor()], dataset_sink_mode=False)

    lora_model.merge_lora_weights()

    lora_model.save_model(save_path)

if __name__ == "__main__":
    finetune_llama3_with_lora('/home/ma-user/work/LLM-FT/ruozhiba_gpt4', '/home/ma-user/work/LLM-FT/01ai/Yi-6B-Chat', '/home/ma-user/work/LLM-FT/01ai/Yi-6B-Chat', 'finetuned.ckpt')
