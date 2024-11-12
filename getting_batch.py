import os
import json
from PIL import Image
from torch.utils.data import Dataset
import re
from torchvision import transforms
import pandas as pd
import numpy as np
import lightning as L
import torch
from typing import Tuple
from transformers import AutoTokenizer, CLIPVisionModel
from torch import nn
from collections import OrderedDict
from transformers.activations import ACT2FN

from lit_llama.tokenizer import Tokenizer
from pathlib import Path
from typing import Optional
import torch.nn as nn
from transformers import AutoModelForCausalLM


def pre_question(question, max_ques_words):
    question = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        question.lower(),
    ).replace(' \t', ' ').replace('is/are', 'is').replace('near/in', 'in')
    question = question.replace('>', 'more than ').replace('-yes/no', '')
    question = question.replace('x ray', 'xray').replace('x-ray', 'xray')
    question = question.rstrip(' ')

    # truncate question
    question_words = question.split(' ')
    if len(question_words) > max_ques_words:
        question = ' '.join(question_words[:max_ques_words])

    return question

# process answer when input model
def pre_answer(answer):
    answer = str(answer)
    answer = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        answer.lower(),
    ).replace(' \t', ' ')
    answer = answer.replace('x ray', 'xray').replace('x-ray', 'xray')
    answer = answer.replace(' - ', '-')
    return answer

def visual_feature(model, image, projector):
    inputs_img = {'pixel_values':0}
    inputs_img['pixel_values'] = image.unsqueeze(0)
    
    #image feature
    image_features = model(**inputs_img)
    selected_image_features = image_features.last_hidden_state[:, 1:]

    #feature embedding
    img_emd = projector(selected_image_features)
    return img_emd
    

class LlavaMultiModalProjector(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.linear_1 = nn.Linear(config["hidden_size"], config["hidden_size_text"], bias=True)
        self.act = ACT2FN[config["projector_hidden_act"]]
        self.linear_2 = nn.Linear(config["hidden_size_text"], config["hidden_size_text"], bias=True)

    def forward(self, image_features):
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states

def tokenize(tokenizer: Tokenizer, string: str, max_length: int, eos=True) -> torch.Tensor:
    return tokenizer.encode(string, bos=True, eos=eos, max_length=max_length)


def generate_prompt_qa(item, content):
    """Generates a standardized message to prompt the model with an instruction, optional input and a
    'response' field."""
    if item == "question": 
        return f"Context:\nImage\nQuestion:\n{content}\nAnswer:\n"
    else:
        return f"Answer:\n{content}"


class vqa_dataset(Dataset):
    def __init__(self, ann_file, transform, vqa_root, eos='[SEP]', max_ques_words=100, max_length = 512,
                 answer_list='', clip_model = None, feature_proj=None, tokenizer = None):
        
        self.ann =  pd.read_pickle(ann_file)

        self.transform = transform
        self.vqa_root = vqa_root
        self.max_ques_words = max_ques_words
        self.eos = eos

        self.clip_model = clip_model
        self.feature_proj = feature_proj
        self.tokenizer = tokenizer
        self.max_length = max_length


    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]

        image_path = os.path.join(self.vqa_root, ann['image'])
        image = Image.open(image_path + '.jpg').convert('RGB')
        image = self.transform(image).cuda()
        
        image_proj = visual_feature(self.clip_model, image, self.feature_proj)
        question = pre_question(ann['question'], self.max_ques_words)
        answers = ann['answer']
        answers = pre_answer(answers)

        full_prompt = generate_prompt_qa('question',question)
        full_prompt_and_response = full_prompt + answers
        encoded_full_prompt = tokenize(self.tokenizer, full_prompt, max_length=self.max_length, eos=False)
        encoded_full_prompt_and_response = tokenize(self.tokenizer, full_prompt_and_response, eos=True, max_length=self.max_length)
        label = encoded_full_prompt_and_response.clone()
        label[:len(encoded_full_prompt)] = -1
        
        
        if len(encoded_full_prompt_and_response)==self.max_length:
            return None

        return {'context':image_proj, 'question':question, 'answer':answers, 'qa_token': encoded_full_prompt_and_response, 'tokens': full_prompt_and_response, 'q_token': encoded_full_prompt, 'label': label}

def prepare_sample(example: dict, max_length: int, mask_inputs: bool = True, config = None):

    ix = torch.randint(len(example), (config["micro_batch_size"],))
    # print("ix", ix)

    input_ids = [example[i]["qa_token"].type(torch.int64) for i in ix]
    # input_ids = torch.stack([example[i]["qa_token"].type(torch.int64) for i in ix])
    labels = [example[i]["label"] for i in ix]
    # print('input_ids', len(input_ids), len(labels))

    max_len = max(len(s) for s in input_ids)
    # print('max_len', max_len)

    def pad_right(x, pad_id):
        # pad right based on the longest sequence
        n = max_len - len(x)
        return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

    x = torch.stack([pad_right(x, pad_id=0) for x in input_ids])
    y = torch.stack([pad_right(y, pad_id=-1) for y in labels])
    # print("xy", x.size(), y.size())

    wte=nn.Embedding(config['padded_vocab_size'], config['n_embd'])
    text_ebd = wte(x)
    # print("emb_x", text_ebd.size())

    # image_features = [example[i]["context"] for i in ix]
    image_features = torch.stack([example[i]["context"].squeeze(0) for i in ix])
    # print("image_features", image_features.size(), type(image_features))

    num_images, num_image_patches, embed_dim = image_features.shape
    # print(num_images, num_image_patches, embed_dim)

    batch_size, sequence_length = x.shape
    # print(batch_size, sequence_length)

    special_image_token_mask = x == config["image_token_index"]
    # print("x", x)
    # print("special_image_token_mask", special_image_token_mask)

    num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
    # print("num_special_image_tokens", num_special_image_tokens)

    max_embed_dim = (num_special_image_tokens.max() * (num_image_patches - 1)) + x.size(1)
    # print("max_embed_dim", max_embed_dim)

    batch_indices, non_image_indices = torch.where(x != config["image_token_index"])
    # print("batch_indices, non_image_indices", batch_indices, non_image_indices)

    new_token_positions = torch.cumsum((special_image_token_mask * (num_image_patches - 1) + 1), -1) - 1
    # print("new_token_positions", new_token_positions)

    nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
    nb_image_pad = nb_image_pad.cuda()
    # print("nb_image_pad", nb_image_pad)
    
    text_to_overwrite = new_token_positions[batch_indices, non_image_indices]
    # print("text_to_overwrite", text_to_overwrite.size())

    final_embedding = torch.zeros(
        batch_size, max_embed_dim, embed_dim, dtype=image_features.dtype, device=image_features.device)

    final_embedding[batch_indices, text_to_overwrite] = text_ebd[batch_indices, non_image_indices].cuda()
    # print('final_embedding[batch_indices, text_to_overwrite]', final_embedding[batch_indices, text_to_overwrite].size())

    image_to_overwrite = torch.all(final_embedding == 0, dim=-1)
    # print("image_to_overwrite", image_to_overwrite.size())
    
    image_to_overwrite &= image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[:, None]
    # print("image_to_overwrite1", image_to_overwrite)

    if image_to_overwrite.sum() != image_features.shape[:-1].numel():
        raise ValueError(
            f"The input provided to the model are wrong. The number of image tokens is {torch.sum(special_image_token_mask)} while"
            f" the number of image given to the model is {num_images}. This prevents correct indexing and breaks batch generation."
        )

    final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim)
    # print("final_embedding[image_to_overwrite]", final_embedding[image_to_overwrite].size())

    # print("final_embedding", final_embedding.size())

    #getting labels
    temp_ebd_label = torch.zeros(batch_size, max_embed_dim, device=image_features.device).long() - 1
    # print("temp_ebd_label", temp_ebd_label.size())

    labels = []
    for i in range(len(ix)):
        ids = ix[i]
        # print("ids", ids)
        temp_len = len(example[ids]["label"])
        temp_ebd_label[i][-temp_len:] = example[ids]["label"]
        labels.append(temp_ebd_label[i])
    labels_ = torch.stack([y for y in labels])

    # return { "input_ids": final_embedding, "labels": y}
    return final_embedding,  labels_


def main():

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
    image_path = '/data/user-data/sa25729/MICCAI_2024/datasets/pvqa/images/train'

    config = {}
    config["hidden_size"] = 768
    config["hidden_size_text"] = 4096
    config["projector_hidden_act"] = 'gelu'
    config["padded_vocab_size"]= 32000
    config["n_embd"] = 4096
    config["micro_batch_size"] = 5
    config["image_token_index"] = 9895 

    model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16").cuda()
    model.eval()
    projector = LlavaMultiModalProjector(config).cuda()

    tokenizer_path = Path("checkpoints/lit-llama/tokenizer.model")
    tokenizer = Tokenizer(tokenizer_path)

    train_dataset = vqa_dataset(train_path, train_transform, image_path, split='train', clip_model = model, feature_proj = projector, tokenizer = tokenizer)

    input_ids, targets = prepare_sample(example = train_dataset, max_length=512, mask_inputs= True, config = config)

    # print('sample_set', input_ids, targets)
    
    

if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main)



