# coding=utf-8
# Copyright Studio-Ouisa and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization classes for LUKE."""

import collections
import copy
import json
import os
from typing import Dict, List, Optional, Tuple, Union

from transformers.models.bert_japanese.tokenization_bert_japanese import (
    BasicTokenizer,
    CharacterTokenizer,
    JumanppTokenizer,
    MecabTokenizer,
    SentencepieceTokenizer,
    SudachiTokenizer,
    WordpieceTokenizer,
    load_vocab,
)
from transformers.models.luke import LukeTokenizer
from transformers.tokenization_utils_base import (
    AddedToken, BatchEncoding, EncodedInput, PaddingStrategy, TensorType, TruncationStrategy
)
from transformers.utils import logging


logger = logging.get_logger(__name__)

EntitySpan = Tuple[int, int]
EntitySpanInput = List[EntitySpan]
Entity = str
EntityInput = List[Entity]

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "entity_vocab_file": "entity_vocab.json"}

PRETRAINED_VOCAB_FILES_MAP = {"vocab_file": {}, "entity_vocab_file": {}}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {}


class LukeBertJapaneseTokenizer(LukeTokenizer):
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask", "position_ids"]

    def __init__(
        self,
        vocab_file,
        entity_vocab_file,
        spm_file=None,
        task=None,
        max_entity_length=32,
        max_mention_length=30,
        entity_token_1="<ent>",
        entity_token_2="<ent2>",
        entity_unk_token="[UNK]",
        entity_pad_token="[PAD]",
        entity_mask_token="[MASK]",
        entity_mask2_token="[MASK2]",
        do_lower_case=False,
        do_word_tokenize=True,
        do_subword_tokenize=True,
        word_tokenizer_type="basic",
        subword_tokenizer_type="wordpiece",
        never_split=None,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        mecab_kwargs=None,
        sudachi_kwargs=None,
        jumanpp_kwargs=None,
        **kwargs,
    ):
        if subword_tokenizer_type == "sentencepiece":
            if not os.path.isfile(spm_file):
                raise ValueError(
                    f"Can't find a vocabulary file at path '{spm_file}'. To load the vocabulary from a Google"
                    " pretrained model use `tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
                )
            self.spm_file = spm_file
        else:
            if not os.path.isfile(vocab_file):
                raise ValueError(
                    f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google"
                    " pretrained model use `tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
                )
            self.vocab = load_vocab(vocab_file)
            self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])

        self.do_word_tokenize = do_word_tokenize
        self.word_tokenizer_type = word_tokenizer_type
        self.lower_case = do_lower_case
        self.never_split = never_split
        self.mecab_kwargs = copy.deepcopy(mecab_kwargs)
        self.sudachi_kwargs = copy.deepcopy(sudachi_kwargs)
        self.jumanpp_kwargs = copy.deepcopy(jumanpp_kwargs)
        if do_word_tokenize:
            if word_tokenizer_type == "basic":
                self.word_tokenizer = BasicTokenizer(
                    do_lower_case=do_lower_case, never_split=never_split, tokenize_chinese_chars=False
                )
            elif word_tokenizer_type == "mecab":
                self.word_tokenizer = MecabTokenizer(
                    do_lower_case=do_lower_case, never_split=never_split, **(mecab_kwargs or {})
                )
            elif word_tokenizer_type == "sudachi":
                self.word_tokenizer = SudachiTokenizer(
                    do_lower_case=do_lower_case, never_split=never_split, **(sudachi_kwargs or {})
                )
            elif word_tokenizer_type == "jumanpp":
                self.word_tokenizer = JumanppTokenizer(
                    do_lower_case=do_lower_case, never_split=never_split, **(jumanpp_kwargs or {})
                )
            else:
                raise ValueError(f"Invalid word_tokenizer_type '{word_tokenizer_type}' is specified.")

        self.do_subword_tokenize = do_subword_tokenize
        self.subword_tokenizer_type = subword_tokenizer_type
        if do_subword_tokenize:
            if subword_tokenizer_type == "wordpiece":
                self.subword_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=str(unk_token))
            elif subword_tokenizer_type == "character":
                self.subword_tokenizer = CharacterTokenizer(vocab=self.vocab, unk_token=str(unk_token))
            elif subword_tokenizer_type == "sentencepiece":
                self.subword_tokenizer = SentencepieceTokenizer(vocab=self.spm_file, unk_token=str(unk_token))
            else:
                raise ValueError(f"Invalid subword_tokenizer_type '{subword_tokenizer_type}' is specified.")

        # we add 2 special tokens for downstream tasks
        # for more information about lstrip and rstrip, see https://github.com/huggingface/transformers/pull/2778
        entity_token_1 = (
            AddedToken(entity_token_1, lstrip=False, rstrip=False)
            if isinstance(entity_token_1, str)
            else entity_token_1
        )
        entity_token_2 = (
            AddedToken(entity_token_2, lstrip=False, rstrip=False)
            if isinstance(entity_token_2, str)
            else entity_token_2
        )
        kwargs["additional_special_tokens"] = kwargs.get("additional_special_tokens", [])
        kwargs["additional_special_tokens"] += [entity_token_1, entity_token_2]

        with open(entity_vocab_file, encoding="utf-8") as entity_vocab_handle:
            self.entity_vocab = json.load(entity_vocab_handle)
        for entity_special_token in [entity_unk_token, entity_pad_token, entity_mask_token, entity_mask2_token]:
            if entity_special_token not in self.entity_vocab:
                raise ValueError(
                    f"Specified entity special token ``{entity_special_token}`` is not found in entity_vocab. "
                    f"Probably an incorrect entity vocab file is loaded: {entity_vocab_file}."
                )
        self.entity_unk_token_id = self.entity_vocab[entity_unk_token]
        self.entity_pad_token_id = self.entity_vocab[entity_pad_token]
        self.entity_mask_token_id = self.entity_vocab[entity_mask_token]
        self.entity_mask2_token_id = self.entity_vocab[entity_mask2_token]

        self.task = task
        if task is None or task == "entity_span_classification":
            self.max_entity_length = max_entity_length
        elif task == "entity_classification":
            self.max_entity_length = 1
        elif task == "entity_pair_classification":
            self.max_entity_length = 2
        else:
            raise ValueError(
                f"Task {task} not supported. Select task from ['entity_classification', 'entity_pair_classification',"
                " 'entity_span_classification'] only."
            )

        self.max_mention_length = max_mention_length

        # We call the grandparent's init, not the parent's.
        super(LukeTokenizer, self).__init__(
            spm_file=spm_file,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            do_lower_case=do_lower_case,
            do_word_tokenize=do_word_tokenize,
            do_subword_tokenize=do_subword_tokenize,
            word_tokenizer_type=word_tokenizer_type,
            subword_tokenizer_type=subword_tokenizer_type,
            never_split=never_split,
            mecab_kwargs=mecab_kwargs,
            sudachi_kwargs=sudachi_kwargs,
            jumanpp_kwargs=jumanpp_kwargs,
            task=task,
            max_entity_length=32,
            max_mention_length=30,
            entity_token_1="<ent>",
            entity_token_2="<ent2>",
            entity_unk_token=entity_unk_token,
            entity_pad_token=entity_pad_token,
            entity_mask_token=entity_mask_token,
            entity_mask2_token=entity_mask2_token,
            **kwargs,
        )

    @property
    # Copied from BertJapaneseTokenizer
    def do_lower_case(self):
        return self.lower_case

    # Copied from BertJapaneseTokenizer
    def __getstate__(self):
        state = dict(self.__dict__)
        if self.word_tokenizer_type in ["mecab", "sudachi", "jumanpp"]:
            del state["word_tokenizer"]
        return state

    # Copied from BertJapaneseTokenizer
    def __setstate__(self, state):
        self.__dict__ = state
        if self.word_tokenizer_type == "mecab":
            self.word_tokenizer = MecabTokenizer(
                do_lower_case=self.do_lower_case, never_split=self.never_split, **(self.mecab_kwargs or {})
            )
        elif self.word_tokenizer_type == "sudachi":
            self.word_tokenizer = SudachiTokenizer(
                do_lower_case=self.do_lower_case, never_split=self.never_split, **(self.sudachi_kwargs or {})
            )
        elif self.word_tokenizer_type == "jumanpp":
            self.word_tokenizer = JumanppTokenizer(
                do_lower_case=self.do_lower_case, never_split=self.never_split, **(self.jumanpp_kwargs or {})
            )

    # Copied from BertJapaneseTokenizer
    def _tokenize(self, text):
        if self.do_word_tokenize:
            tokens = self.word_tokenizer.tokenize(text, never_split=self.all_special_tokens)
        else:
            tokens = [text]

        if self.do_subword_tokenize:
            split_tokens = [sub_token for token in tokens for sub_token in self.subword_tokenizer.tokenize(token)]
        else:
            split_tokens = tokens

        return split_tokens

    @property
    # Copied from BertJapaneseTokenizer
    def vocab_size(self):
        if self.subword_tokenizer_type == "sentencepiece":
            return len(self.subword_tokenizer.sp_model)
        return len(self.vocab)

    # Copied from BertJapaneseTokenizer
    def get_vocab(self):
        if self.subword_tokenizer_type == "sentencepiece":
            vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
            vocab.update(self.added_tokens_encoder)
            return vocab
        return dict(self.vocab, **self.added_tokens_encoder)

    # Copied from BertJapaneseTokenizer
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        if self.subword_tokenizer_type == "sentencepiece":
            return self.subword_tokenizer.sp_model.PieceToId(token)
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    # Copied from BertJapaneseTokenizer
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if self.subword_tokenizer_type == "sentencepiece":
            return self.subword_tokenizer.sp_model.IdToPiece(index)
        return self.ids_to_tokens.get(index, self.unk_token)

    # Copied from BertJapaneseTokenizer
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        if self.subword_tokenizer_type == "sentencepiece":
            return self.subword_tokenizer.sp_model.decode(tokens)
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    # Copied from BertJapaneseTokenizer
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A BERT sequence has the following format:
        - single sequence: `[CLS] X [SEP]`
        - pair of sequences: `[CLS] A [SEP] B [SEP]`
        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    # Copied from BertJapaneseTokenizer
    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.
        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.
        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    # Copied from BertJapaneseTokenizer
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A BERT sequence
        pair mask has the following format:
        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```
        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).
        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    # Copied and modified from LukeTokenizer, removing the `add_prefix_space` process
    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        return (text, kwargs)

    # Copied and modified from LukeTokenizer, adding `position_ids` to the output
    def prepare_for_model(
        self,
        ids: List[int],
        pair_ids: Optional[List[int]] = None,
        entity_ids: Optional[List[int]] = None,
        pair_entity_ids: Optional[List[int]] = None,
        entity_token_spans: Optional[List[Tuple[int, int]]] = None,
        pair_entity_token_spans: Optional[List[Tuple[int, int]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        max_entity_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        prepend_batch_axis: bool = False,
        **kwargs,
    ) -> BatchEncoding:
        """
        Prepares a sequence of input id, entity id and entity span, or a pair of sequences of inputs ids, entity ids,
        entity spans so that it can be used by the model. It adds special tokens, truncates sequences if overflowing
        while taking into account the special tokens and manages a moving window (with user defined stride) for
        overflowing tokens. Please Note, for *pair_ids* different than `None` and *truncation_strategy = longest_first*
        or `True`, it is not possible to return overflowing tokens. Such a combination of arguments will raise an
        error.

        Args:
            ids (`List[int]`):
                Tokenized input ids of the first sequence.
            pair_ids (`List[int]`, *optional*):
                Tokenized input ids of the second sequence.
            entity_ids (`List[int]`, *optional*):
                Entity ids of the first sequence.
            pair_entity_ids (`List[int]`, *optional*):
                Entity ids of the second sequence.
            entity_token_spans (`List[Tuple[int, int]]`, *optional*):
                Entity spans of the first sequence.
            pair_entity_token_spans (`List[Tuple[int, int]]`, *optional*):
                Entity spans of the second sequence.
            max_entity_length (`int`, *optional*):
                The maximum length of the entity sequence.
        """

        # Backward compatibility for 'truncation_strategy', 'pad_to_max_length'
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        # Compute lengths
        pair = bool(pair_ids is not None)
        len_ids = len(ids)
        len_pair_ids = len(pair_ids) if pair else 0

        if return_token_type_ids and not add_special_tokens:
            raise ValueError(
                "Asking to return token_type_ids while setting add_special_tokens to False "
                "results in an undefined behavior. Please set add_special_tokens to True or "
                "set return_token_type_ids to None."
            )
        if (
            return_overflowing_tokens
            and truncation_strategy == TruncationStrategy.LONGEST_FIRST
            and pair_ids is not None
        ):
            raise ValueError(
                "Not possible to return overflowing tokens for pair of sequences with the "
                "`longest_first`. Please select another truncation strategy than `longest_first`, "
                "for instance `only_second` or `only_first`."
            )

        # Load from model defaults
        if return_token_type_ids is None:
            return_token_type_ids = "token_type_ids" in self.model_input_names
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        encoded_inputs = {}

        # Compute the total size of the returned word encodings
        total_len = len_ids + len_pair_ids + (self.num_special_tokens_to_add(pair=pair) if add_special_tokens else 0)

        # Truncation: Handle max sequence length and max_entity_length
        overflowing_tokens = []
        if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE and max_length and total_len > max_length:
            # truncate words up to max_length
            ids, pair_ids, overflowing_tokens = self.truncate_sequences(
                ids,
                pair_ids=pair_ids,
                num_tokens_to_remove=total_len - max_length,
                truncation_strategy=truncation_strategy,
                stride=stride,
            )

        if return_overflowing_tokens:
            encoded_inputs["overflowing_tokens"] = overflowing_tokens
            encoded_inputs["num_truncated_tokens"] = total_len - max_length

        # Add special tokens
        if add_special_tokens:
            sequence = self.build_inputs_with_special_tokens(ids, pair_ids)
            token_type_ids = self.create_token_type_ids_from_sequences(ids, pair_ids)
            entity_token_offset = 1  # 1 * <s> token
            pair_entity_token_offset = len(ids) + 3  # 1 * <s> token & 2 * <sep> tokens
        else:
            sequence = ids + pair_ids if pair else ids
            token_type_ids = [0] * len(ids) + ([0] * len(pair_ids) if pair else [])
            entity_token_offset = 0
            pair_entity_token_offset = len(ids)

        # Build output dictionary
        encoded_inputs["input_ids"] = sequence
        encoded_inputs["position_ids"] = list(range(len(sequence)))
        if return_token_type_ids:
            encoded_inputs["token_type_ids"] = token_type_ids
        if return_special_tokens_mask:
            if add_special_tokens:
                encoded_inputs["special_tokens_mask"] = self.get_special_tokens_mask(ids, pair_ids)
            else:
                encoded_inputs["special_tokens_mask"] = [0] * len(sequence)

        # Set max entity length
        if not max_entity_length:
            max_entity_length = self.max_entity_length

        if entity_ids is not None:
            total_entity_len = 0
            num_invalid_entities = 0
            valid_entity_ids = [ent_id for ent_id, span in zip(entity_ids, entity_token_spans) if span[1] <= len(ids)]
            valid_entity_token_spans = [span for span in entity_token_spans if span[1] <= len(ids)]

            total_entity_len += len(valid_entity_ids)
            num_invalid_entities += len(entity_ids) - len(valid_entity_ids)

            valid_pair_entity_ids, valid_pair_entity_token_spans = None, None
            if pair_entity_ids is not None:
                valid_pair_entity_ids = [
                    ent_id
                    for ent_id, span in zip(pair_entity_ids, pair_entity_token_spans)
                    if span[1] <= len(pair_ids)
                ]
                valid_pair_entity_token_spans = [span for span in pair_entity_token_spans if span[1] <= len(pair_ids)]
                total_entity_len += len(valid_pair_entity_ids)
                num_invalid_entities += len(pair_entity_ids) - len(valid_pair_entity_ids)

            if num_invalid_entities != 0:
                logger.warning(
                    f"{num_invalid_entities} entities are ignored because their entity spans are invalid due to the"
                    " truncation of input tokens"
                )

            if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE and total_entity_len > max_entity_length:
                # truncate entities up to max_entity_length
                valid_entity_ids, valid_pair_entity_ids, overflowing_entities = self.truncate_sequences(
                    valid_entity_ids,
                    pair_ids=valid_pair_entity_ids,
                    num_tokens_to_remove=total_entity_len - max_entity_length,
                    truncation_strategy=truncation_strategy,
                    stride=stride,
                )
                valid_entity_token_spans = valid_entity_token_spans[: len(valid_entity_ids)]
                if valid_pair_entity_token_spans is not None:
                    valid_pair_entity_token_spans = valid_pair_entity_token_spans[: len(valid_pair_entity_ids)]

            if return_overflowing_tokens:
                encoded_inputs["overflowing_entities"] = overflowing_entities
                encoded_inputs["num_truncated_entities"] = total_entity_len - max_entity_length

            final_entity_ids = valid_entity_ids + valid_pair_entity_ids if valid_pair_entity_ids else valid_entity_ids
            encoded_inputs["entity_ids"] = list(final_entity_ids)
            entity_position_ids = []
            entity_start_positions = []
            entity_end_positions = []
            for token_spans, offset in (
                (valid_entity_token_spans, entity_token_offset),
                (valid_pair_entity_token_spans, pair_entity_token_offset),
            ):
                if token_spans is not None:
                    for start, end in token_spans:
                        start += offset
                        end += offset
                        position_ids = list(range(start, end))[: self.max_mention_length]
                        position_ids += [-1] * (self.max_mention_length - end + start)
                        entity_position_ids.append(position_ids)
                        entity_start_positions.append(start)
                        entity_end_positions.append(end - 1)

            encoded_inputs["entity_position_ids"] = entity_position_ids
            if self.task == "entity_span_classification":
                encoded_inputs["entity_start_positions"] = entity_start_positions
                encoded_inputs["entity_end_positions"] = entity_end_positions

            if return_token_type_ids:
                encoded_inputs["entity_token_type_ids"] = [0] * len(encoded_inputs["entity_ids"])

        # Check lengths
        self._eventual_warn_about_too_long_sequence(encoded_inputs["input_ids"], max_length, verbose)

        # Padding
        if padding_strategy != PaddingStrategy.DO_NOT_PAD or return_attention_mask:
            encoded_inputs = self.pad(
                encoded_inputs,
                max_length=max_length,
                max_entity_length=max_entity_length,
                padding=padding_strategy.value,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )

        if return_length:
            encoded_inputs["length"] = len(encoded_inputs["input_ids"])

        batch_outputs = BatchEncoding(
            encoded_inputs, tensor_type=return_tensors, prepend_batch_axis=prepend_batch_axis
        )

        return batch_outputs

    # Copied and modified from LukeTokenizer, adding the padding of `position_ids`
    def _pad(
        self,
        encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
        max_length: Optional[int] = None,
        max_entity_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
    ) -> dict:
        """
        Pad encoded inputs (on left/right and up to predefined length or max length in the batch)


        Args:
            encoded_inputs:
                Dictionary of tokenized inputs (`List[int]`) or batch of tokenized inputs (`List[List[int]]`).
            max_length: maximum length of the returned list and optionally padding length (see below).
                Will truncate by taking into account the special tokens.
            max_entity_length: The maximum length of the entity sequence.
            padding_strategy: PaddingStrategy to use for padding.


                - PaddingStrategy.LONGEST Pad to the longest sequence in the batch
                - PaddingStrategy.MAX_LENGTH: Pad to the max length (default)
                - PaddingStrategy.DO_NOT_PAD: Do not pad
                The tokenizer padding sides are defined in self.padding_side:


                    - 'left': pads on the left of the sequences
                    - 'right': pads on the right of the sequences
            pad_to_multiple_of: (optional) Integer if set will pad the sequence to a multiple of the provided value.
                This is especially useful to enable the use of Tensor Core on NVIDIA hardware with compute capability
                `>= 7.5` (Volta).
            return_attention_mask:
                (optional) Set to False to avoid returning attention mask (default: set to model specifics)
        """
        entities_provided = bool("entity_ids" in encoded_inputs)

        # Load from model defaults
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(encoded_inputs["input_ids"])
            if entities_provided:
                max_entity_length = len(encoded_inputs["entity_ids"])

        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        if (
            entities_provided
            and max_entity_length is not None
            and pad_to_multiple_of is not None
            and (max_entity_length % pad_to_multiple_of != 0)
        ):
            max_entity_length = ((max_entity_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and (
            len(encoded_inputs["input_ids"]) != max_length
            or (entities_provided and len(encoded_inputs["entity_ids"]) != max_entity_length)
        )

        # Initialize attention mask if not present.
        if return_attention_mask and "attention_mask" not in encoded_inputs:
            encoded_inputs["attention_mask"] = [1] * len(encoded_inputs["input_ids"])
        if entities_provided and return_attention_mask and "entity_attention_mask" not in encoded_inputs:
            encoded_inputs["entity_attention_mask"] = [1] * len(encoded_inputs["entity_ids"])

        if needs_to_be_padded:
            difference = max_length - len(encoded_inputs["input_ids"])
            if entities_provided:
                entity_difference = max_entity_length - len(encoded_inputs["entity_ids"])
            if self.padding_side == "right":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = encoded_inputs["attention_mask"] + [0] * difference
                    if entities_provided:
                        encoded_inputs["entity_attention_mask"] = (
                            encoded_inputs["entity_attention_mask"] + [0] * entity_difference
                        )
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] = encoded_inputs["token_type_ids"] + [0] * difference
                    if entities_provided:
                        encoded_inputs["entity_token_type_ids"] = (
                            encoded_inputs["entity_token_type_ids"] + [0] * entity_difference
                        )
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = encoded_inputs["special_tokens_mask"] + [1] * difference
                encoded_inputs["input_ids"] = encoded_inputs["input_ids"] + [self.pad_token_id] * difference
                encoded_inputs["position_ids"] = encoded_inputs["position_ids"] + [0] * difference
                if entities_provided:
                    encoded_inputs["entity_ids"] = (
                        encoded_inputs["entity_ids"] + [self.entity_pad_token_id] * entity_difference
                    )
                    encoded_inputs["entity_position_ids"] = (
                        encoded_inputs["entity_position_ids"] + [[-1] * self.max_mention_length] * entity_difference
                    )
                    if self.task == "entity_span_classification":
                        encoded_inputs["entity_start_positions"] = (
                            encoded_inputs["entity_start_positions"] + [0] * entity_difference
                        )
                        encoded_inputs["entity_end_positions"] = (
                            encoded_inputs["entity_end_positions"] + [0] * entity_difference
                        )

            elif self.padding_side == "left":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [0] * difference + encoded_inputs["attention_mask"]
                    if entities_provided:
                        encoded_inputs["entity_attention_mask"] = [0] * entity_difference + encoded_inputs[
                            "entity_attention_mask"
                        ]
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] = [0] * difference + encoded_inputs["token_type_ids"]
                    if entities_provided:
                        encoded_inputs["entity_token_type_ids"] = [0] * entity_difference + encoded_inputs[
                            "entity_token_type_ids"
                        ]
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = [1] * difference + encoded_inputs["special_tokens_mask"]
                encoded_inputs["input_ids"] = [self.pad_token_id] * difference + encoded_inputs["input_ids"]
                encoded_inputs["position_ids"] = [0] * difference + encoded_inputs["position_ids"]
                if entities_provided:
                    encoded_inputs["entity_ids"] = [self.entity_pad_token_id] * entity_difference + encoded_inputs[
                        "entity_ids"
                    ]
                    encoded_inputs["entity_position_ids"] = [
                        [-1] * self.max_mention_length
                    ] * entity_difference + encoded_inputs["entity_position_ids"]
                    if self.task == "entity_span_classification":
                        encoded_inputs["entity_start_positions"] = [0] * entity_difference + encoded_inputs[
                            "entity_start_positions"
                        ]
                        encoded_inputs["entity_end_positions"] = [0] * entity_difference + encoded_inputs[
                            "entity_end_positions"
                        ]
            else:
                raise ValueError("Invalid padding strategy:" + str(self.padding_side))

        return encoded_inputs

    # Copied and modified from BertJapaneseTokenizer and LukeTokenizer
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if os.path.isdir(save_directory):
            if self.subword_tokenizer_type == "sentencepiece":
                vocab_file = os.path.join(
                    save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["spm_file"]
                )
            else:
                vocab_file = os.path.join(
                    save_directory,
                    (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"],
                )
        else:
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory

        if self.subword_tokenizer_type == "sentencepiece":
            with open(vocab_file, "wb") as writer:
                content_spiece_model = self.subword_tokenizer.sp_model.serialized_model_proto()
                writer.write(content_spiece_model)
        else:
            with open(vocab_file, "w", encoding="utf-8") as writer:
                index = 0
                for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                    if index != token_index:
                        logger.warning(
                            f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                            " Please check that the vocabulary is not corrupted!"
                        )
                        index = token_index
                    writer.write(token + "\n")
                    index += 1

        entity_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["entity_vocab_file"]
        )

        with open(entity_vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.entity_vocab, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        return vocab_file, entity_vocab_file
