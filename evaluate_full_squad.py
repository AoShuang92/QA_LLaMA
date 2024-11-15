import math
import json
import sys
import time
from pathlib import Path
from typing import Optional
import logging
from nltk.translate.bleu_score import sentence_bleu
import lightning as L
import torch
import tqdm
import evaluate

from lit_llama import LLaMA, Tokenizer
from lit_llama.utils import EmptyInitOnDevice, lazy_load, llama_model_lookup
from lit_llama.lora import lora
from generate import generate
from scripts.prepare_squad import generate_prompt_qa


from datasets import load_dataset

lora_r = 8
lora_alpha = 16
lora_dropout = 0.05
max_seq_length = 512

rouge = evaluate.load('rouge', keep_in_memory=True)
exact_match_metric = evaluate.load("exact_match")
meteor = evaluate.load('meteor')

def normalize_text(s):
  """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
  import string, re

  def remove_articles(text):
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    return re.sub(regex, " ", text)

  def white_space_fix(text):
    return " ".join(text.split())

  def remove_punc(text):
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)

  def lower(text):
    return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact_match(prediction, truth):
    return int(normalize_text(prediction) == normalize_text(truth))

def compute_f1(prediction, truth):
  pred_tokens = normalize_text(prediction).split()
  truth_tokens = normalize_text(truth).split()
  
  # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
  if len(pred_tokens) == 0 or len(truth_tokens) == 0:
    return int(pred_tokens == truth_tokens)
  
  common_tokens = set(pred_tokens) & set(truth_tokens)
  
  # if there are no common tokens then f1 = 0
  if len(common_tokens) == 0:
    return 0
  
  prec = len(common_tokens) / len(pred_tokens)
  rec = len(common_tokens) / len(truth_tokens)
  
  return 2 * (prec * rec) / (prec + rec)


def model_predict(context, question, model, tokenizer):

    full_prompt = generate_prompt_qa({"context":context,"question":question})
    encoded = tokenizer.encode(full_prompt, bos=True, eos=False, device=model.device)
    prompt_input_len = len(encoded)
    with torch.no_grad():
        output, prob = generate( 
            model,
            idx=encoded,
            max_seq_length=max_seq_length,
            max_new_tokens=100,
            temperature=1,
            eos_id=tokenizer.eos_id,
            argmax=True
        )
    return tokenizer.decode(output[prompt_input_len:-1]), prob


def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)
    logging.getLogger().addHandler(logging.StreamHandler()) # print to terminal also (logging.info)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')      
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

@torch.no_grad()
def main(
    squad_dataset_path: Optional[Path] = None,
    *,
    accelerator: str = "auto",
    checkpoint_path: Optional[Path] = None,
    tokenizer_path: Optional[Path] = None,
    model_size: str = "7B",
    dtype: str = "bfloat16",
    quantize: Optional[str] = None,
) -> None:
    """Generates text samples based on a pre-trained LLaMA model and tokenizer
       finetuned with LoRA.

    Args:
        datasets: The datasets to use as a comma separated string
        # compile: Whether to compile the model.
        accelerator: The hardware to run on. Possible choices are:
            ``"cpu"``, ``"cuda"``, ``"mps"``, ``"gpu"``, ``"tpu"``, ``"auto"``.
        lora_path: Path to the checkpoint with trained LoRA weights, which are the output of
            `finetune_lora.py`.
        checkpoint_path: The checkpoint path to load.
        tokenizer_path: The tokenizer path to load.
        quantize: Whether to quantize the model and using which method:
            ``"llm.int8"``: LLM.int8() mode,
            ``"gptq.int4"``: GPTQ 4-bit mode.
    """
    # print('checkpoint_path', checkpoint_path) None
    if not checkpoint_path:
        checkpoint_path = Path(f"./checkpoints/lit-llama/7B/lit-llama.pth")
    if not tokenizer_path:
        tokenizer_path = Path("./checkpoints/lit-llama/tokenizer.model")
    if not squad_dataset_path:
        squad_dataset_path = Path("./generation_qa/squad2.0/dev-v2.0.json")

    assert checkpoint_path.is_file()
    assert tokenizer_path.is_file()
    assert squad_dataset_path.is_file()
    # print('squad_dataset_path', squad_dataset_path) generation_qa/squad2.0/dev-v2.0.json
    with open(squad_dataset_path, "rb") as f:
        squad_eval_data = json.load(f)
        

    fabric = L.Fabric(accelerator=accelerator, devices=1)
    fabric.seed_everything(42)

    dt = getattr(torch, dtype, None)
    if not isinstance(dt, torch.dtype):
        raise ValueError(f"{dtype} is not a valid dtype.")
    dtype = dt

    print("Loading model ...", file=sys.stderr)
    log_name = "".join(str(checkpoint_path).split("/")[-1].split(".")[:-1]) + "-only-argmax"
    log_path =  checkpoint_path.parent
    t0 = time.time()
    logger = setup_logger("Squad Evaluation Log", Path(log_path / f"squad_evaluation_{log_name}.log"))

    with EmptyInitOnDevice(
        device=fabric.device, dtype=dtype, quantization_mode=quantize
    ):
        print("Loading model ...", file=sys.stderr)
        t0 = time.time()
        model = LLaMA.from_name(model_size)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint)
        print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)
    
    model.eval()

    # if compile:
    #     model = torch.compile(model)
    model = fabric.setup_module(model)

    tokenizer = Tokenizer(tokenizer_path)
    
    t0 = time.time() # reset 

    predict_dict = {}
    prob_dict = {}
    total_count= 0
    unk_count = 0
    unk_gt_count = 0
    f1_count = 0
    em_score = 0
    f1_score = 0
    blue_score1 = 0
    # Search for each passage, its question and its answer
    for group in squad_eval_data["data"]:
        for passage in group["paragraphs"]:
            context = passage["context"]
            for qa in passage["qas"]:
                total_count+=1
                question = qa["question"]
                try:
                    gt_answer = qa["answers"][0]["text"]
                except:
                    gt_answer = ""
                    unk_gt_count+=1
                special_id = qa["id"]
                model_answer, model_prob = model_predict(context, question, model, tokenizer)

                if model_answer == "<unk>":
                    unk_count+=1
                    model_answer = ""

                predict_dict[special_id] = model_answer
                prob_dict[special_id] = model_prob
                logger.info(f"{total_count} -- Predicted Answer: {model_answer}  |||  Ground Truth: {gt_answer}  ||| Prob: {model_prob:0.4f}")

                model_answer_ = model_answer.split("\n\n")[0]
                # model_answer_ = model_answer
                gt_answer = gt_answer
                # .split(' ')

                # print('comparison:', gt_answer, model_answer_, type(gt_answer), type(model_answer_), len(gt_answer), len(model_answer_))
                # BLEU_1 = sentence_bleu(gt_answer, model_answer_, weights=(1, 0, 0, 0))
                # print('BLEU_1', BLEU_1)

                # f1_ans = compute_f1(model_answer_,gt_answer)
                # print('f1_ans', f1_ans)

                # em_score += compute_exact_match(model_answer_, gt_answer)
                # print('em_score', em_score)

                meteor_results = meteor.compute(predictions=[model_answer_], references=[gt_answer]) 
                # print('meteor_results', meteor_results['meteor'])

                # ignore <unk> for now in terms of f1_score (use EM to gauge performance there)
                if gt_answer != "" and model_answer_ != "":
                    f1_score += compute_f1(model_answer_,gt_answer)
                    em_score += compute_exact_match(model_answer_, gt_answer)
                    blue_score1 += sentence_bleu(gt_answer, model_answer_, weights=(1, 0, 0, 0))
                    
                    f1_count+=1

                if total_count % 500 == 0:
                    logger.info(f"F1 Score: {f1_score/f1_count }\n")
                    logger.info(f"EM Score: {em_score/f1_count }\n")
                    logger.info(f"BLEU1 Score: {blue_score1/f1_count }\n")

    logger.info(f"Total unknown prediction:{unk_count} , total unknown ground truth: {unk_gt_count}\n")
    logger.info(f"Total F1 Score: {f1_score/f1_count * 100}\n")
    logger.info(f"Total EM Score: {em_score/total_count * 100}\n")
    logger.info(f"Total BLEU Score: {blue_score1/total_count * 100}\n")
    logger.info(f"Inference through whole dataset: {time.time() - t0:.02f} seconds.\n")

    # Save predictions for further evaluation by official script (EM and F1)
    with open(Path(log_path / f"squad_eval_predict_{log_name}.json"), "w") as fp:
        json.dump(predict_dict, fp)
    with open(Path(log_path / f"squad_eval_probs_{log_name}.json"), "w") as fp:
        json.dump(prob_dict, fp)

    logger.info(
        f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB"
    )


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    CLI(main)
