from glob import glob
import json
import math
import os
import random
import numpy as np
import torch
import torch.utils.data
import random
from tqdm import tqdm
import re
import torch.nn.functional as F

import languages 


class CharTokenizer:
    """Trivial tokenizer – just use UTF-8 bytes"""
    eot = 0
    
    def encode(self, txt):
        return list(bytes(txt.strip(), 'utf-8'))

    def decode(self, tokens):
        return bytes(tokens).decode('utf-8')


def char_per_seconder(semantic_tokens, text, stoks_per_second=25):
    secs = semantic_tokens.shape[-1] / stoks_per_second
    cps = len(text) / secs
    return cps


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, val=False):
        if val:
            datafile = cfg['data']['validation_data']
        else:
            datafile = cfg['data']['training_data']
        with open(datafile) as f:
            self.data = f.read().splitlines()
        self.semantic_dir = cfg['data']['semantic_dir']
        self.tokenizer = CharTokenizer()
        self.ttoks_len = int(cfg['data'].get("max_text_len", 480))
        self.stoks_len = int(cfg['data'].get("max_semantic_len", 2020))
        self.stoks_codes = int(cfg['data'].get("num_codes", 1024))
        self.weight = 1
        random.shuffle(self.data)


    def get_data(self, item):
        try:
            audiofilepath, text, speaker, language = item.split("|")
        except Exception as e:
            if len(item.split("|")) != 4:
                splits = item.split("|")
                audiofilepath, speaker, language = splits[0], splits[-2], splits[-1]
                text = ". ".join(splits[1:-2])
        try:
            _, filename = audiofilepath.split("/", maxsplit=1)

            text = text.strip()
            tokenized_text = torch.tensor(self.tokenizer.encode(text))
            tokenized_text = F.pad(tokenized_text, (0, self.ttoks_len - tokenized_text.shape[-1]), value=self.tokenizer.eot)

            semantic_path = os.path.join(self.semantic_dir, filename.replace(".flac", ".wav").replace(".mp3", ".wav").replace(".wav", ".wav.tok.pt"))
            semantic_tokens = torch.load(semantic_path).squeeze(0)

            cps = char_per_seconder(semantic_tokens, text)
            language = languages.to_id(language)

            return tokenized_text.detach(), semantic_tokens.detach(), cps, language
        except Exception as e:
            return None


    def __getitem__(self, index):
        return self.get_data(self.data[index])


    def __len__(self):
        return len(self.data)



class TextAudioCollate:
    def __init__(self):
        self.tokenizer = CharTokenizer()
        self.stoks_len = 2020
        self.stoks_codes = 1024
        self.ttoks_len = 480

    def __call__(self, batch):
        batch = [b for b in batch if b is not None]
        
        toks = [x[0] for x in batch]
        toks = torch.stack(toks, dim=0)
        if isinstance(toks, (list, np.ndarray)): toks = torch.tensor(toks)
        toks = toks.to(torch.long)
        in_ttoks = F.pad(toks, (1, self.ttoks_len - toks.shape[-1] - 1), value=self.tokenizer.eot)
        out_ttoks = F.pad(toks, (0, self.ttoks_len - toks.shape[-1]), value=self.tokenizer.eot)

        stoks_ = [x[1].to(torch.long) for x in batch]
        # stoks = torch.stack(stoks, dim=0)
        # if isinstance(stoks, (list, np.ndarray)): stoks = torch.tensor(stoks)
        # stoks = stoks.to(torch.long)
        # in_stoks = F.pad(stoks, (1, self.stoks_len - stoks.shape[-1] - 1), value=self.stoks_codes-1)
        # out_stoks = F.pad(stoks, (0, self.stoks_len - stoks.shape[-1]), value=self.stoks_codes-1)

        in_stoks = torch.stack([F.pad(stoks, (1, self.stoks_len - stoks.shape[-1] - 1), value=self.stoks_codes-1) for stoks in stoks_], dim=0)
        out_stoks = torch.stack([F.pad(stoks, (0, self.stoks_len - stoks.shape[-1]), value=self.stoks_codes-1) for stoks in stoks_], dim=0)

        cps = torch.tensor([x[2] for x in batch])
        lang = torch.tensor([x[3] for x in batch])

        return in_ttoks, out_ttoks, lang, cps, in_stoks, out_stoks
        # return {
        #     "in_ttoks": in_ttoks,
        #     "out_ttoks": out_ttoks,
        #     "language": lang,
        #     "cps": cps,
        #     "in_stoks": in_stoks,
        #     "out_stoks": out_stoks,
        # }