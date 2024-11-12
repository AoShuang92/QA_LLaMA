"""
Instruction-tuning with LoRA on the Alpaca dataset.

Note: If you run into a CUDA error "Expected is_sm80 to be true, but got false", uncomment the line
`torch.backends.cuda.enable_flash_sdp(False)` in the script below (see https://github.com/Lightning-AI/lit-llama/issues/101).
"""
import os
import time
import json
import logging
from PIL import Image
import lightning as L
import numpy as np
import torch
from pathlib import Path
from datasets import load_dataset
from generate import generate
from lit_llama.lora import mark_only_lora_as_trainable, lora, lora_state_dict
from lit_llama.model import LLaMA, LLaMAConfig
from lit_llama.tokenizer import Tokenizer
from scripts.prepare_alpaca import generate_prompt
from getting_batch import prepare_sample
from getting_batch import vqa_dataset
from getting_batch import LlavaMultiModalProjector

from transformers import AutoTokenizer, CLIPVisionModel
from torch import nn
from collections import OrderedDict
from transformers.activations import ACT2FN
from torchvision import transforms


VALIDATE_PERCENTAGE = 0.8
eval_interval = 100
log_loss_iters = 100
eval_iters = 2500
log_interval = 1000
devices = 2
# Hyperparameters
learning_rate = 3e-5
batch_size = 128 // devices
micro_batch_size = 5
gradient_accumulation_steps = batch_size // micro_batch_size
epoch = 5
training_data_size = 12000
max_iters = (training_data_size * epoch) // devices // micro_batch_size
weight_decay = 0.0
max_seq_length = 512  
lora_r = 16
lora_alpha = 16
lora_dropout = 0.05
warmup_steps = 100

# Setting up logger

def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')      
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

def main(
    
    data_dir: str = "data/squad2",
    pretrained_path: str = "checkpoints/lit-llama/7B/lit-llama.pth",
    out_dir: str = "out/lora/squad2_r16_new_loss",
):

    fabric = L.Fabric(accelerator="cuda", devices=devices, precision="bf16-true",strategy="ddp")
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    
    # train_data, val_data = load_datasets(data_dir=data_dir)
    # print("train_data", train_data[0])

    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    train_transform = transforms.Compose([
        # transforms.RandomResizedCrop(384, scale=(0.5, 1.0), interpolation=Image.BICUBIC),
        transforms.Resize((224,224), interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        # RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
        #                                       'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        transforms.ToTensor(),
        normalize,
    ])


    train_path = '/data/user-data/sa25729/MICCAI_2024/datasets/pvqa/qas/train/train_qa.pkl'
    val_path = '/data/user-data/sa25729/MICCAI_2024/datasets/pvqa/qas/val/val_qa.pkl'
    test_path = '/data/user-data/sa25729/MICCAI_2024/datasets/pvqa/qas/test/test_qa.pkl'

    train_image_path = '/data/user-data/sa25729/MICCAI_2024/datasets/pvqa/images/train'
    val_image_path = '/data/user-data/sa25729/MICCAI_2024/datasets/pvqa/images/val'
    test_image_path = '/data/user-data/sa25729/MICCAI_2024/datasets/pvqa/images/test'


    configs = {}
    configs["hidden_size"] = 768
    configs["hidden_size_text"] = 4096
    configs["projector_hidden_act"] = 'gelu'
    configs["padded_vocab_size"]= 32000
    configs["n_embd"] = 4096
    configs["micro_batch_size"] = 4
    configs["image_token_index"] = 9895 

    model_clip = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16").cuda()
    model_clip.eval()
    projector = LlavaMultiModalProjector(configs).cuda()

    tokenizer_path = Path("checkpoints/lit-llama/tokenizer.model")
    tokenizer = Tokenizer(tokenizer_path)

    train_dataset = vqa_dataset(train_path, train_transform, train_image_path, clip_model = model_clip, feature_proj = projector, tokenizer = tokenizer)
    val_dataset = vqa_dataset(val_path, train_transform, val_image_path, clip_model = model_clip, feature_proj = projector, tokenizer = tokenizer)
    test_dataset = vqa_dataset(test_path, train_transform, test_image_path, clip_model = model_clip, feature_proj = projector, tokenizer = tokenizer)

    config = LLaMAConfig.from_name("7B")
    config.block_size = max_seq_length

    checkpoint = torch.load(pretrained_path)

    with fabric.init_module(), lora(r=lora_r, alpha=lora_alpha, dropout=lora_dropout, enabled=True):
        model = LLaMA(config)
        # strict=False because missing keys due to LoRA weights not contained in checkpoint state
        model.load_state_dict(checkpoint, strict=False)
    
    mark_only_lora_as_trainable(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model, optimizer = fabric.setup(model, optimizer)
    train(fabric, model, optimizer, train_dataset, val_dataset, out_dir, configs)

    # Save the final LoRA checkpoint at the end of training
    checkpoint = lora_state_dict(model)
    fabric.save(os.path.join(out_dir, "lit-llama-lora-finetuned.pth"), checkpoint)


def train(
    fabric: L.Fabric,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: np.ndarray,
    val_data: np.ndarray,
    out_dir: str,
    configs
) -> None:
    """The training loop.

    Loosely based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT.
    """
    step_count = 0
    prev_val_loss = 9999
    train_loss = 0
    train_loss_logger = setup_logger("Train Loss Log", os.path.join(out_dir,"train_loss.log"))
    val_loss_logger = setup_logger("Val Loss Log", os.path.join(out_dir,"val_loss.log"))
    validate_output_logger = setup_logger("Val output log", os.path.join(out_dir,"val_out.log"))
    
    for iter_num in range(max_iters):

        if step_count <= warmup_steps:
            # linear warmup
            lr = learning_rate * step_count / warmup_steps

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        t0 = time.time()

        is_accumulating = (iter_num + 1) % gradient_accumulation_steps != 0
    
        # input_ids, targets = get_batch_(fabric, train_data)
        input_ids, targets = prepare_sample(example = train_data, max_length=512, mask_inputs= True, config = configs)
         
        with fabric.no_backward_sync(model,enabled=is_accumulating):
            # print('input_ids', input_ids.size())
            logits = model(input_ids)

            #verify the content of input_ids
            # tokenizer_path: Path = Path("checkpoints/lit-llama/tokenizer.model")
            # tokenizer = Tokenizer(tokenizer_path)
            # dec= tokenizer.decode(input_ids)
            # dec_ans= tokenizer.decode(targets)
            # print('decoding',dec)
            # print('decoding answers ',dec_ans)

            # print('input_ids1', input_ids.size())  
            # print('logits', logits.size()) 
            # print('targets', targets.size()) 
            # logits torch.Size([1, 155, 32000])
            # targets torch.Size([1, 155])
            # print('logits', logits.size())
            # print('targets', targets.size()) 
            loss = loss_fn(logits, targets)
            train_loss += loss
            fabric.backward(loss)
        # print("is_accumulating", is_accumulating) 

        # if not is_accumulating:
        if is_accumulating:
            optimizer.step()
            optimizer.zero_grad()
            step_count += 1

            # print("1",step_count % log_loss_iters, fabric.global_rank) 
            if step_count % log_loss_iters == 0 and fabric.global_rank == 0:
                train_loss_logger.info(f"step: {step_count}, training loss: {train_loss}")

            # print("2",step_count % eval_interval )   
            if step_count % eval_interval == 0:
                val_loss = validate_all_data(fabric, model, val_data, logger = validate_output_logger, configs= configs)
                if fabric.global_rank==0:
                    val_loss_logger.info(f"step: {step_count}, val loss: {val_loss}")
                fabric.print(f"step {iter_num}: val loss {val_loss:.4f}")
                fabric.barrier()
            # print("3", step_count % eval_interval)
            if step_count % eval_interval == 0: #and val_loss < prev_val_loss:
                prev_val_loss = val_loss 
                # print(f"Saving LoRA weights to {out_dir}")
                # We are only saving the LoRA weights
                # TODO: Provide a function/script to merge the LoRA weights with pretrained weights
                checkpoint = lora_state_dict(model)
                fabric.save(os.path.join(out_dir, f"iter-{iter_num}-loss-{val_loss:.4f}-ckpt.pth"), checkpoint)
                fabric.barrier()
            
            train_loss = 0 # reset the train loss (accumulate and reset)

        dt = time.time() - t0
        if iter_num % log_interval == 0:
            fabric.print(f"iter {iter_num}: loss {loss.item():.4f}, time: {dt*1000:.2f}ms")


def generate_response(model, instruction):
    tokenizer = Tokenizer("checkpoints/lit-llama/tokenizer.model")
    sample = {"instruction": instruction, "input": ""}
    prompt = generate_prompt(sample)
    encoded = tokenizer.encode(prompt, bos=True, eos=False, device=model.device)

    output = generate(
        model,
        idx=encoded,
        max_seq_length=max_seq_length,
        max_new_tokens=100,
    )
    output = tokenizer.decode(output)
    return output # output.split("### Response:")[1].strip()

@torch.no_grad()
def validate_all_data(fabric: L.Fabric, model: torch.nn.Module, val_data: np.ndarray, logger, configs) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()
    ceiling_division = lambda a,b : -(a//-b)
    data_portion = int(len(val_data)*VALIDATE_PERCENTAGE)
    data_splits = ceiling_division(data_portion, devices)
    all_losses_main = []
    for i in range(devices):
        if fabric.global_rank == i:
            start = i*data_splits
            end = (i+1)*data_splits
            data_chunk = val_data

            # get the loss for the data on device
            all_loss = []
            # input_ids = []
            # labels = []
            


            for count, data in enumerate(data_chunk,1):
                    
                input_ids, targets = prepare_sample(example = val_data, max_length=512, mask_inputs= True, config = configs)
            
            #     # stack up to micro batch size
            #     input_ids.append(data["input_ids"].type(torch.int64))
            #     labels.append(data["labels"].type(torch.int64))

            #     if count % micro_batch_size == 0:
            #         max_len = max(len(s) for s in input_ids)


            #         def pad_right(x, pad_id):
            #             # pad right based on the longest sequence
            #             n = max_len - len(x)
            #             return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

            #         x = torch.stack([pad_right(x, pad_id=0) for x in input_ids])
            #         y = torch.stack([pad_right(x, pad_id=-1) for x in labels])
            #         x, y = fabric.to_device((x.pin_memory(), y.pin_memory()))

                # print("input_ids val", input_ids.size())
                logits = model(input_ids)
                loss = loss_fn(logits, targets)
                all_loss.append(loss.item())

                # reset input and output
                # input_ids  = []
                # labels = []
            
            # every device will add the mean loss here:
            all_losses_main.append(sum(all_loss) / len(all_loss))
    
    # ---- wait for all devices to complete (use fabric barrier) -----------
    fabric.barrier()
    losses = fabric.all_gather(all_losses_main)[0]
    assert losses.shape[0] == devices, "Each device will calculate the mean of loss. The current amount of loss data does not match amount of devices "
    model.train()
    return losses.mean().item()

def loss_fn(logits, targets):
    # shift the targets such that output n predicts token n+1
    logits = logits[..., :-1, :].contiguous() #remove the last position token eos (2)
    
    targets = targets[..., 1:].contiguous() # remove the bos token (1)

    #reasons for the removing and contiguous: answer prediction ending at eos token

    # print('logits_contiguous', logits.size())
    # print('targets_contiguous', targets.size())
    # logits_contiguous torch.Size([1, 154, 32000])
    # targets_contiguous torch.Size([1, 154])

    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

    #ignore_index=-1: targets are embedding as the same size of input_ids, and the padding positions are with value -1.
    #logit is the size of input_ids, ignore the index of -1, meaning only compare the predicted ans and targets for loss calcuation.

    return loss

def get_batch_(fabric: L.Fabric, data: list):
    ix = torch.randint(len(data), (micro_batch_size,))
    # print('ix', ix) 

    input_ids = data[ix]["input_ids"]
    labels = data[ix]["labels"].long().cuda()

    return input_ids, labels

    

def get_batch(fabric: L.Fabric, data: list):
    ix = torch.randint(len(data), (micro_batch_size,))
    print('ix', ix) 
    

    input_ids = [data[i]["input_ids"].type(torch.int64) for i in ix]
    labels = [data[i]["labels"].type(torch.int64) for i in ix]


    # print('input_ids',len(input_ids) ,input_ids[0].size(), input_ids[1].size(), input_ids[2].size(), input_ids[3].size(), input_ids[4].size())
    # print('labels', len(labels),labels[0].size(), labels[1].size(), labels[2].size(), labels[3].size(), labels[4].size())
    
    # print('input_ids',len(input_ids) ,input_ids[0].size()) 1, [155]
    # print('labels', len(labels),labels[0].size()) 1, [155]

    # print('input_ids',input_ids ,input_ids[0].size()) 
    # print('labels', labels,labels[0].size())

    max_len = max(len(s) for s in input_ids)
    print('max_len', max_len)

    def pad_right(x, pad_id):
        # pad right based on the longest sequence
        n = max_len - len(x)
        return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))
        
    
    x = torch.stack([pad_right(x, pad_id=0) for x in input_ids])
    y = torch.stack([pad_right(x, pad_id=-1) for x in labels])
    x, y = fabric.to_device((x.pin_memory(), y.pin_memory()))

    # print('x',len(x) ,x[0].size(), x[1].size(), x[2].size(), x[3].size(), x[4].size())
    # print('y', len(y),y[0].size(), y[1].size(), y[2].size(), y[3].size(), y[4].size())
    # print("visalisation", x[0], y[0]) # padding at right, input_ids (context+q+a) padding id is 0, answer padding id is -1.
    return x, y



def load_datasets(data_dir):
    train_data = torch.load(os.path.join(data_dir, "train_data_fuse_v1.pt"))
    val_data = torch.load(os.path.join(data_dir, "train_data_fuse_v1.pt"))
    # train_data = torch.load(os.path.join(data_dir, "train.pt"))
    # val_data = torch.load(os.path.join(data_dir, "test.pt"))
    return train_data, val_data
    


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")
    
    from jsonargparse.cli import CLI
    CLI(main)
