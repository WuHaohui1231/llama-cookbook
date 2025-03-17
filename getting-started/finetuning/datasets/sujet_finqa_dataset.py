# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 3 Community License Agreement.


import copy
from datasets import load_dataset
import itertools
import torch
import json

# check system prompt token seq or user prompt token seq is in the current token list
def check_header(targets,seq):
    for i in range(len(seq)-3):
        if seq[i:i+3] in targets:
            return True
    return False
def replace_target(target,seq):
    for i in range(len(seq)-3):
        if seq[i:i+3] == target:
            seq[i],seq[i+1],seq[i+2] = -100,-100,-100
    return seq
def tokenize_dialogs(dialogs, images, processor):
    text_prompt = processor.apply_chat_template(dialogs)
    batch = processor(images=images, text=text_prompt,padding = True, return_tensors="pt")
    label_list = []
    for i in range(len(batch["input_ids"])):
        dialog_tokens = batch["input_ids"][i].tolist()
        labels = copy.copy(dialog_tokens)
        eot_indices = [i for i,n in enumerate(labels) if n == 128009]
        last_idx = 0
        # system prompt header "<|start_header_id|>system<|end_header_id|>" has been tokenized to [128006, 9125, 128007]
        # user prompt header "<|start_header_id|>user<|end_header_id|>" has been tokenized to [128006, 882, 128007]
        prompt_header_seqs = [[128006, 9125, 128007],[128006, 882, 128007]]
        for n, idx in enumerate(eot_indices):
            current_seq = labels[last_idx:idx+1]
            if check_header(prompt_header_seqs,current_seq):
                # found prompt header, indicating that this seq should be masked
                labels[last_idx:idx+1] = [-100] * (idx-last_idx+1)
            else:
                last_idx = idx+1
            #  Mask all the assistant header prompt <|start_header_id|>assistant<|end_header_id|>, which has been tokenized to [128006, 78191, 128007]
        assistant_header_seq = [128006, 78191, 128007]
        labels = replace_target(assistant_header_seq,labels)
        # Mask the padding token and image token 128256 
        for i in range(len(labels)):
            if labels[i] == processor.tokenizer.pad_token_id or labels[i] == 128256: #  128256 is image token index
                labels[i] = -100
        label_list.append(labels)
    batch["labels"] = torch.tensor(label_list)
    return batch


def get_custom_dataset(dataset_config, processor, split, split_ratio=0.9):
    # load_dataset will return DatasetDict that contains all the data in the train set
    dataset_dict = load_dataset("sujet-ai/Sujet-Finance-QA-Vision-100k")
    dataset = dataset_dict['train']
    # Comment out the following line to use the full dataset, for quick testing only use 2000 samples
    # dataset = dataset.select(range(2000))
    dataset = dataset.train_test_split(test_size=1-split_ratio, shuffle=True, seed=42)[split]
    return dataset

class SujetFQADataCollator:
    def __init__(self, processor):
        self.processor = processor
        self.processor.tokenizer.padding_side = "right" # during training, one always uses padding on the right
    def __call__(self, samples):
        dialogs,images = [],[]
        for sample in samples:
            image = sample["image"]
            sample_list = json.loads(sample["qa_pairs"])
            # if len(image_list) > 1:
            #     raise ValueError("Only support one image per sample")
            image = image.convert("RGB") # only use the first image
            dialog = []
            for sample_dict in sample_list:
                if not dialog:
                    # only append image to the first sentence
                    dialog += [
                    {"role":"user","content":[{"type": "image"},{"type": "text", "text": sample_dict["question"].strip()}]},
                    {"role":"assistant","content":[{"type": "text", "text": sample_dict["answer"].strip()}]}
                ]
                
                else:
                    dialog += [
                    {"role":"user","content":[{"type": "text", "text": sample_dict["question"].strip()}]},
                    {"role":"assistant","content":[{"type": "text", "text": sample_dict["answer"].strip()}]}
                ]
            dialogs.append(dialog)
            images.append([image])
        return tokenize_dialogs(dialogs,images, self.processor)
def get_data_collator(processor):
    return SujetFQADataCollator(processor)
