import torch
from transformers import AutoTokenizer
import os


def install_tokenizer(path: str = "meta-llama/Llama-2-7b-chat-hf"):
    out_path = os.getcwd() + "/tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True, force_download=True)
    special_tokens_dict = {'additional_special_tokens': ['<user>', '<bot>']}
    tokenizer.add_special_tokens(special_tokens_dict)
    tokenizer.save_pretrained(out_path)


def get_tokenizer():
    model_path = os.getcwd() + "/tokenizer"
    tokenizer =  AutoTokenizer.from_pretrained(model_path, use_fast=True)
    return tokenizer
