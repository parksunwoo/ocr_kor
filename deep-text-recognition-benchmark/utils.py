import torch
import json
from typing import List, Callable, Union

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CTCLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'blank' token required by CTCLoss
            self.dict[char] = i + 1

        self.character = ['[blank]'] + dict_character  # dummy '[blank]' token for CTCLoss (index 0)

    def encode(self, text, batch_max_length=25):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
        output:
            text: concatenated text index for CTCLoss.
                    [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]
        text = ''.join(text)
        text = [self.dict[char] for char in text]

        return (torch.IntTensor(text).to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        index = 0
        for l in length:
            t = text_index[index:index + l]

            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)

            texts.append(text)
            index += l
        return texts


class AttnLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        list_token = ['[GO]', '[s]']  # ['[s]','[UNK]','[PAD]','[GO]']
        list_character = list(character)
        self.character = list_token + list_character

        self.dict = {}
        for i, char in enumerate(self.character):
            # print(i, char)
            self.dict[char] = i

    def encode(self, text, batch_max_length=25):
        """ convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default
        output:
            text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
            length : the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """
        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
        # batch_max_length = max(length) # this is not allowed for multi-gpu setting
        batch_max_length += 1
        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text.append('[s]')
            text = [self.dict[char] for char in text]
            batch_text[i][1:1 + len(text)] = torch.LongTensor(text)  # batch_text[:, 0] = [GO] token
        return (batch_text.to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts



class Averager(object):
    """Compute average for torch.Tensor, used for loss average."""

    def __init__(self):
        self.reset()

    def add(self, v):
        count = v.data.numel()
        v = v.data.sum()
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res



class Vocab:
    """Vocab class"""
    def __init__(self, list_of_tokens: List[str] = None, padding_token: str = '<pad>', unknown_token: str = '<unk>',
                bos_token: str = '<bos>', eos_token: str = '<eos>', reserved_tokens: List[str] = None, unknown_token_idx: int = 0):
        """Instantiating Vocab class
        Args:
            list_of_tokens (List[str]): list of tokens is source of vocabulary. each token is not duplicate
            padding_token (str): the representation for padding token
            unknown_token (str): the representation for any unknown token
            bos_token (str): the representation for the special token of beginning-of-sequence token
            eos_token (str): the representation for the special token of end-of-sequence token
            reserved_tokens (List[str]): a list specifying additional tokens to be added to the vocabulary
            unknown_token_idx (int): the specific integer is mapped to unknown token
        """
        self._unknown_token = unknown_token
        self._padding_token = padding_token
        self._bos_token = bos_token
        self._eos_token = eos_token
        self._reserved_tokens = reserved_tokens
        self._special_tokens = []

        for tkn in [self._padding_token, self._bos_token, self._eos_token]:
            if tkn:
                self._special_tokens.append(tkn)

        if self._reserved_tokens:
            self._special_tokens.extend(self._reserved_tokens)
        if self._unknown_token:
            self._special_tokens.insert(unknown_token_idx, self._unknown_token)

        if list_of_tokens:
            self._special_tokens.extend(list(filter(lambda elm: elm not in self._special_tokens, list_of_tokens)))

        self._token_to_idx, self._idx_to_token = self._build(self._special_tokens)
        self._embedding = None

    def to_indices(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        """Looks up indices of text tokens according to the vocabulary
        Args:
            tokens (Union[str, List[str]]): a source token or tokens to be converted
        Returns:
            Union[int, List[int]]: a token index or a list of token indices according to the vocabulary
        """
        if isinstance(tokens, list):
            return [self._token_to_idx[tkn] if tkn in self._token_to_idx else self._token_to_idx[self._unknown_token]
                    for tkn in tokens]
        else:
            return self._token_to_idx[tokens] if tokens in self._token_to_idx else \
                self._token_to_idx[self._unknown_token]

    def to_tokens(self, indices: Union[int, List[int]]) -> Union[str, List[str]]:
        """Converts token indices to tokens according to the vocabulary
        Args:
            indices (Union[int, List[int]]): a source token index or token indices to be converted
        Returns:
            Union[str, List[str]]: a token or a list of tokens according to the vocabulary
        """
        if isinstance(indices, list):
            return [self._idx_to_token[idx] for idx in indices]
        else:
            return self._idx_to_token[indices]

    def _build(self, list_of_tokens):
        token_to_idx = {tkn: idx for idx, tkn in enumerate(list_of_tokens)}
        idx_to_token = {idx: tkn for idx, tkn in enumerate(list_of_tokens)}
        return token_to_idx, idx_to_token

    def __len__(self):
        return len(self._token_to_idx)

    @property
    def token_to_idx(self):
        return self._token_to_idx

    @property
    def idx_to_token(self):
        return self._idx_to_token

    @property
    def padding_token(self):
        return self._padding_token

    @property
    def unknown_token(self):
        return self._unknown_token

    @property
    def bos_token(self):
        return self._bos_token

    @property
    def eos_token(self):
        return self._eos_token

    @property
    def embedding(self):
        return self._embedding

    @embedding.setter
    def embedding(self, array):
        self._embedding = array



class Tokenizer:
    """Tokenizer class"""
    def __init__(self, vocab: Vocab, split_fn: Callable[[str], List[str]], pad_fn: Callable[[List[int]], List[int]] = None) -> None:
        """Instantiating Tokenizer class

        Args:
            vocab (model.utils.Vocab): the instance of model.utils.Vocab created from specific split_fn
            split_fn (Callable): a function that can act as a splitter
            pad_fn (Callable): a function that can act as a padder
        """
        self._vocab = vocab
        self._split = split_fn
        self._pad = pad_fn

    def split(self, string: str) -> List[str]:
        list_of_tokens = self._split(string)
        return list_of_tokens

    def transform(self, list_of_tokens: List[str]) -> List[int]:
        list_of_indices = self._vocab.to_indices(list_of_tokens)
        list_of_indices = self._pad(list_of_indices) if self._pad else list_of_indices
        return list_of_indices

    def split_and_transform(self, string: str) -> List[int]:
        return self.transform(self.split(string))

    @property
    def vocab(self):
        return self._vocab

class PadSequence:
    """PadSequence class"""
    def __init__(self, length: int, pad_val: int = 0, clip: bool = True) -> None:
        """Instantiating PadSequence class
        Args:
            length (int): the maximum length to pad/clip the sequence
            pad_val (int): the pad value
            clip (bool): whether to clip the length, if sample length is longer than maximum length
        """
        self._length = length
        self._pad_val = pad_val
        self._clip = clip

    def __call__(self, sample):
        sample_length = len(sample)
        if sample_length >= self._length:
            if self._clip and sample_length > self._length:
                return sample[:self._length]
            else:
                return sample
        else:
            return sample + [self._pad_val for _ in range(self._length - sample_length)]



class PreProcessor(Tokenizer):
    def preprocess(self, string):
        list_of_tokens = self.split(string)
        if len(list_of_tokens) >= self._pad._length:
            list_of_tokens = list_of_tokens[:(self._pad._length - 1)]
        list_of_tokens = ['[CLS]'] + list_of_tokens
        list_of_indices = self.transform(list_of_tokens)
        return list_of_indices


class Config:
    def __init__(self, json_path):
        with open(json_path, mode='r') as io:
            params = json.loads(io.read())
        self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, mode='w') as io:
            json.dump(self.__dict__, io, indent=4)

    def update(self, json_path):
        with open(json_path, mode='r') as io:
            params = json.loads(io.read())
        self.__dict__.update(params)

    @property
    def dict(self):
        return self.__dict__
