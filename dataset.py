import re
from collections import Counter
from typing import Dict, List, Tuple, Optional

import spacy
from datasets import load_dataset

def _regex_tokenize(text: str) -> List[str]:
    return re.findall(r'\w+|[^\w\s]', text.lower(), re.UNICODE)


class Multi30kDataset:
    def __init__(
        self,
        split: str = "train",
        min_freq: int = 1,
        max_vocab_size: Optional[int] = None,
        src_vocab: Optional[Dict[str, Dict]] = None,
        tgt_vocab: Optional[Dict[str, Dict]] = None,
    ) -> None:
        """
        Loads the Multi30k dataset and prepares tokenizers.
        """
        self.split = split
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size

        self.dataset = load_dataset("bentrevett/multi30k")
        try:
            self.spacy_de = spacy.load("de_core_news_sm")
        except OSError:
            self.spacy_de = None
        try:
            self.spacy_en = spacy.load("en_core_web_sm")
        except OSError:
            self.spacy_en = None

        self.special_tokens = ["<pad>", "<unk>", "<sos>", "<eos>"]
        self.pad_idx = 0
        self.unk_idx = 1
        self.sos_idx = 2
        self.eos_idx = 3

        if src_vocab is None or tgt_vocab is None:
            self.src_vocab, self.tgt_vocab = self.build_vocab()
        else:
            self.src_vocab = src_vocab
            self.tgt_vocab = tgt_vocab

        self.data = self.process_data()

    def _tokenize_de(self, text: str) -> List[str]:
        if self.spacy_de is not None:
            return [tok.text.lower() for tok in self.spacy_de.tokenizer(text)]
        return _regex_tokenize(text)

    def _tokenize_en(self, text: str) -> List[str]:
        if self.spacy_en is not None:
            return [tok.text.lower() for tok in self.spacy_en.tokenizer(text)]
        return _regex_tokenize(text)

    def _build_vocab_from_counter(self, counter: Counter) -> Dict[str, Dict]:
        tokens = [tok for tok, freq in counter.items() if freq >= self.min_freq]
        tokens.sort(key=lambda t: (-counter[t], t))

        if self.max_vocab_size is not None:
            tokens = tokens[: max(0, self.max_vocab_size - len(self.special_tokens))]

        itos = self.special_tokens + tokens
        stoi = {tok: idx for idx, tok in enumerate(itos)}
        return {"itos": itos, "stoi": stoi}

    def _extract_pair(self, example: dict) -> Tuple[str, str]:
        if "translation" in example:
            return example["translation"]["de"], example["translation"]["en"]
        return example["de"], example["en"]

    def build_vocab(self) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
        """
        Builds the vocabulary mapping for src (de) and tgt (en), including:
        <unk>, <pad>, <sos>, <eos>
        """
        src_counter = Counter()
        tgt_counter = Counter()

        for example in self.dataset["train"]:
            de_text, en_text = self._extract_pair(example)
            src_counter.update(self._tokenize_de(de_text))
            tgt_counter.update(self._tokenize_en(en_text))

        return self._build_vocab_from_counter(src_counter), self._build_vocab_from_counter(tgt_counter)

    def process_data(self) -> List[Tuple[List[int], List[int]]]:
        """
        Convert English and German sentences into integer token lists using
        spacy and the defined vocabulary.
        """
        data = []
        src_stoi = self.src_vocab["stoi"]
        tgt_stoi = self.tgt_vocab["stoi"]

        for example in self.dataset[self.split]:
            de_text, en_text = self._extract_pair(example)
            de_tokens = ["<sos>"] + self._tokenize_de(de_text) + ["<eos>"]
            en_tokens = ["<sos>"] + self._tokenize_en(en_text) + ["<eos>"]

            de_ids = [src_stoi.get(tok, self.unk_idx) for tok in de_tokens]
            en_ids = [tgt_stoi.get(tok, self.unk_idx) for tok in en_tokens]
            data.append((de_ids, en_ids))

        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        return self.data[idx]