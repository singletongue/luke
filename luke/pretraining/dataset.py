import functools
import itertools
import json
import logging
import multiprocessing
import os
import random
import re
from contextlib import closing
from multiprocessing.pool import Pool
from typing import Optional

import click
import tensorflow as tf
import transformers
from tensorflow.io import TFRecordWriter
from tensorflow.train import Int64List
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer
from wikipedia2vec.dump_db import DumpDB

from luke.pretraining.tokenization import tokenize, tokenize_segments
from luke.utils.entity_vocab import UNK_TOKEN, EntityVocab
from luke.utils.model_utils import (
    ENTITY_VOCAB_FILE,
    METADATA_FILE,
    get_entity_vocab_file_path,
)
from luke.utils.sentence_splitter import SentenceSplitter

logger = logging.getLogger(__name__)

DATASET_FILE = "dataset.tf"

# global variables used in pool workers
_dump_db = _tokenizer = _sentence_splitter = _entity_vocab = _max_num_tokens = _max_entity_length = None
_max_mention_length = _min_sentence_length = _include_sentences_without_entities = _include_unk_entities = None
_abstract_only = _language = _add_distantly_supervised_links = _min_distantly_supervised_link_text_length = None


@click.command()
@click.argument("dump_db_file", type=click.Path(exists=True))
@click.argument("tokenizer_name")
@click.argument("entity_vocab_file", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path(file_okay=False))
@click.option("--language", type=str)
@click.option("--sentence-splitter", default="en")
@click.option("--max-seq-length", default=512)
@click.option("--max-entity-length", default=128)
@click.option("--max-mention-length", default=30)
@click.option("--min-sentence-length", default=5)
@click.option("--abstract-only", is_flag=True)
@click.option("--include-sentences-without-entities", is_flag=True)
@click.option("--include-unk-entities/--skip-unk-entities", default=False)
@click.option("--add-distantly-supervised-links", is_flag=True)
@click.option("--min-distantly-supervised-link-text-length", default=2)
@click.option("--pool-size", default=multiprocessing.cpu_count())
@click.option("--chunk-size", default=100)
@click.option("--max-num-documents", default=None, type=int)
@click.option("--predefined-entities-only", is_flag=True)
def build_wikipedia_pretraining_dataset(
    dump_db_file: str,
    tokenizer_name: str,
    entity_vocab_file: str,
    output_dir: str,
    language: Optional[str],
    sentence_splitter: str,
    **kwargs
):
    dump_db = DumpDB(dump_db_file)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
    sentence_splitter = SentenceSplitter.from_name(sentence_splitter)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    entity_vocab = EntityVocab(entity_vocab_file)
    WikipediaPretrainingDataset.build(
        dump_db, tokenizer, sentence_splitter, entity_vocab, output_dir, language, **kwargs
    )


class WikipediaPretrainingDataset:
    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir
        with open(os.path.join(dataset_dir, METADATA_FILE)) as metadata_file:
            self.metadata = json.load(metadata_file)

    def __len__(self):
        return self.metadata["number_of_items"]

    @property
    def max_seq_length(self):
        return self.metadata["max_seq_length"]

    @property
    def max_entity_length(self):
        return self.metadata["max_entity_length"]

    @property
    def max_mention_length(self):
        return self.metadata["max_mention_length"]

    @property
    def language(self):
        return self.metadata.get("language", None)

    @property
    def tokenizer(self):
        tokenizer_class_name = self.metadata.get("tokenizer_class", "")
        tokenizer_class = getattr(transformers, tokenizer_class_name)
        return tokenizer_class.from_pretrained(self.dataset_dir)

    @property
    def entity_vocab(self):
        vocab_file_path = get_entity_vocab_file_path(self.dataset_dir)
        return EntityVocab(vocab_file_path)

    def create_iterator(
        self,
        skip: int = 0,
        num_workers: int = 1,
        worker_index: int = 0,
        shuffle_buffer_size: int = 1000,
        shuffle_seed: int = 0,
        num_parallel_reads: int = 10,
        repeat: bool = True,
        shuffle: bool = True,
    ):

        # The TensorFlow 2.0 has enabled eager execution by default.
        # At the starting of algorithm, we need to use this to disable eager execution.
        tf.compat.v1.disable_eager_execution()

        features = dict(
            word_ids=tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            entity_ids=tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            entity_position_ids=tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            page_id=tf.io.FixedLenFeature([1], tf.int64, default_value=[0]),
        )
        dataset = tf.data.TFRecordDataset(
            [os.path.join(self.dataset_dir, DATASET_FILE)],
            compression_type="GZIP",
            num_parallel_reads=num_parallel_reads,
        )
        if repeat:
            dataset = dataset.repeat()
        if shuffle:
            dataset = dataset.shuffle(shuffle_buffer_size, seed=shuffle_seed)
        dataset = dataset.skip(skip)
        dataset = dataset.shard(num_workers, worker_index)
        dataset = dataset.map(functools.partial(tf.io.parse_single_example, features=features))
        it = tf.compat.v1.data.make_one_shot_iterator(dataset)
        it = it.get_next()

        with tf.compat.v1.Session() as sess:
            try:
                while True:
                    obj = sess.run(it)
                    yield dict(
                        page_id=obj["page_id"][0],
                        word_ids=obj["word_ids"],
                        entity_ids=obj["entity_ids"],
                        entity_position_ids=obj["entity_position_ids"].reshape(-1, self.metadata["max_mention_length"]),
                    )
            except tf.errors.OutOfRangeError:
                pass

    @classmethod
    def build(
        cls,
        dump_db: DumpDB,
        tokenizer: PreTrainedTokenizer,
        sentence_splitter: SentenceSplitter,
        entity_vocab: EntityVocab,
        output_dir: str,
        language: Optional[str],
        max_seq_length: int,
        max_entity_length: int,
        max_mention_length: int,
        min_sentence_length: int,
        abstract_only: bool,
        include_sentences_without_entities: bool,
        include_unk_entities: bool,
        add_distantly_supervised_links: bool,
        min_distantly_supervised_link_text_length: int,
        pool_size: int,
        chunk_size: int,
        max_num_documents: Optional[int],
        predefined_entities_only: bool,
    ):

        target_titles = [
            title
            for title in dump_db.titles()
            if not (":" in title and title.lower().split(":")[0] in ("image", "file", "category"))
        ]

        if predefined_entities_only:
            target_titles = [title for title in target_titles if entity_vocab.contains(title, language)]

        random.shuffle(target_titles)

        if max_num_documents is not None:
            target_titles = target_titles[:max_num_documents]

        max_num_tokens = max_seq_length - 2  # 2 for [CLS] and [SEP]

        tokenizer.save_pretrained(output_dir)

        entity_vocab.save(os.path.join(output_dir, ENTITY_VOCAB_FILE))
        number_of_items = 0
        tf_file = os.path.join(output_dir, DATASET_FILE)
        options = tf.io.TFRecordOptions(tf.compat.v1.io.TFRecordCompressionType.GZIP)
        with TFRecordWriter(tf_file, options=options) as writer:
            with tqdm(total=len(target_titles)) as pbar:
                initargs = (
                    dump_db,
                    tokenizer,
                    sentence_splitter,
                    entity_vocab,
                    language,
                    max_num_tokens,
                    max_entity_length,
                    max_mention_length,
                    min_sentence_length,
                    abstract_only,
                    include_sentences_without_entities,
                    include_unk_entities,
                    add_distantly_supervised_links,
                    min_distantly_supervised_link_text_length,
                )
                with closing(
                    Pool(pool_size, initializer=WikipediaPretrainingDataset._initialize_worker, initargs=initargs)
                ) as pool:
                    for ret in pool.imap(
                        WikipediaPretrainingDataset._process_page, target_titles, chunksize=chunk_size
                    ):
                        for data in ret:
                            writer.write(data)
                            number_of_items += 1
                        pbar.update()

        with open(os.path.join(output_dir, METADATA_FILE), "w") as metadata_file:
            json.dump(
                dict(
                    number_of_items=number_of_items,
                    max_seq_length=max_seq_length,
                    max_entity_length=max_entity_length,
                    max_mention_length=max_mention_length,
                    min_sentence_length=min_sentence_length,
                    tokenizer_class=tokenizer.__class__.__name__,
                    language=language,
                ),
                metadata_file,
                indent=2,
            )

    @staticmethod
    def _initialize_worker(
        dump_db: DumpDB,
        tokenizer: PreTrainedTokenizer,
        sentence_splitter: SentenceSplitter,
        entity_vocab: EntityVocab,
        language: Optional[str],
        max_num_tokens: int,
        max_entity_length: int,
        max_mention_length: int,
        min_sentence_length: int,
        abstract_only: bool,
        include_sentences_without_entities: bool,
        include_unk_entities: bool,
        add_distantly_supervised_links: bool,
        min_distantly_supervised_link_text_length: int,
    ):
        global _dump_db, _tokenizer, _sentence_splitter, _entity_vocab, _max_num_tokens, _max_entity_length
        global _max_mention_length, _min_sentence_length, _include_sentences_without_entities, _include_unk_entities
        global _abstract_only
        global _language
        global _add_distantly_supervised_links
        global _min_distantly_supervised_link_text_length

        _dump_db = dump_db
        _tokenizer = tokenizer
        _sentence_splitter = sentence_splitter
        _entity_vocab = entity_vocab
        _max_num_tokens = max_num_tokens
        _max_entity_length = max_entity_length
        _max_mention_length = max_mention_length
        _min_sentence_length = min_sentence_length
        _include_sentences_without_entities = include_sentences_without_entities
        _include_unk_entities = include_unk_entities
        _abstract_only = abstract_only
        _language = language
        _add_distantly_supervised_links = add_distantly_supervised_links
        _min_distantly_supervised_link_text_length = min_distantly_supervised_link_text_length

    @staticmethod
    def _process_page(page_title: str):
        if _entity_vocab.contains(page_title, _language):
            page_id = _entity_vocab.get_id(page_title, _language)
        else:
            page_id = -1

        sentences = []

        if _add_distantly_supervised_links:
            # Gather all links present in the page
            page_links = []
            for paragraph in _dump_db.get_paragraphs(page_title):
                if _abstract_only and not paragraph.abstract:
                    continue

                for link in paragraph.wiki_links:
                    if len(link.text) < _min_distantly_supervised_link_text_length:
                        continue

                    link_title = _dump_db.resolve_redirect(link.title)
                    if link_title.startswith("Category:") and link.text.lower().startswith("category:"):
                        continue
                    if not _entity_vocab.contains(link_title, _language):
                        continue

                    link = (link.text, link_title)
                    if link in page_links:
                        continue

                    page_links.append(link)

            # Sort the links by text length (longest first)
            page_links = sorted(page_links, key=lambda x: len(x[0]), reverse=True)

        for paragraph in _dump_db.get_paragraphs(page_title):

            if _abstract_only and not paragraph.abstract:
                continue

            paragraph_text = paragraph.text

            # First, get paragraph links.
            # Parapraph links are represented its form (link_title) and the start/end positions of strings
            # (link_start, link_end).
            paragraph_links = []
            for link in paragraph.wiki_links:
                link_title = _dump_db.resolve_redirect(link.title)
                # remove category links
                if link_title.startswith("Category:") and link.text.lower().startswith("category:"):
                    paragraph_text = (
                        paragraph_text[: link.start] + " " * (link.end - link.start) + paragraph_text[link.end :]
                    )
                else:
                    if _entity_vocab.contains(link_title, _language):
                        paragraph_links.append((link_title, link.start, link.end))
                    elif _include_unk_entities:
                        paragraph_links.append((UNK_TOKEN, link.start, link.end))

            if _add_distantly_supervised_links:
                # Add distantly supervised links from the page
                for link_text, link_title in page_links:
                    for match in re.finditer(re.escape(link_text), paragraph_text):
                        ds_start, ds_end = match.start(), match.end()
                        for _, start, end in paragraph_links:
                            # Ignore overlapping links
                            if start < ds_end or ds_start < end:
                                break
                        else:
                            paragraph_links.append((link_title, ds_start, ds_end))

                # Sort the updated links by start position
                paragraph_links = sorted(paragraph_links, key=lambda x: x[1])

            sent_spans = _sentence_splitter.get_sentence_spans(paragraph_text.rstrip())
            for sent_start, sent_end in sent_spans:
                cur = sent_start
                sent_words = []
                sent_links = []
                # Look for links that are within the tokenized sentence.
                # If a link is found, we separate the sentences across the link and tokenize them.
                for link_title, link_start, link_end in paragraph_links:
                    if not (sent_start <= link_start < sent_end and link_end <= sent_end):
                        continue
                    entity_id = _entity_vocab.get_id(link_title, _language)

                    sent_tokenized, link_words = tokenize_segments(
                        [paragraph_text[cur:link_start], paragraph_text[link_start:link_end]],
                        tokenizer=_tokenizer,
                        add_prefix_space=cur == 0 or paragraph_text[cur - 1] == " ",
                    )

                    sent_words += sent_tokenized

                    sent_links.append((entity_id, len(sent_words), len(sent_words) + len(link_words)))
                    sent_words += link_words
                    cur = link_end

                sent_words += tokenize(
                    text=paragraph_text[cur:sent_end],
                    tokenizer=_tokenizer,
                    add_prefix_space=cur == 0 or paragraph_text[cur - 1] == " ",
                )

                if len(sent_words) < _min_sentence_length or len(sent_words) > _max_num_tokens:
                    continue
                sentences.append((sent_words, sent_links))

        ret = []
        words = []
        links = []
        for i, (sent_words, sent_links) in enumerate(sentences):
            links += [(id_, start + len(words), end + len(words)) for id_, start, end in sent_links]
            words += sent_words
            if i == len(sentences) - 1 or len(words) + len(sentences[i + 1][0]) > _max_num_tokens:
                if links or _include_sentences_without_entities:
                    links = links[:_max_entity_length]
                    word_ids = _tokenizer.convert_tokens_to_ids(words)
                    assert _min_sentence_length <= len(word_ids) <= _max_num_tokens
                    entity_ids = [id_ for id_, _, _, in links]
                    assert len(entity_ids) <= _max_entity_length
                    entity_position_ids = itertools.chain(
                        *[
                            (list(range(start, end)) + [-1] * (_max_mention_length - end + start))[:_max_mention_length]
                            for _, start, end in links
                        ]
                    )

                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature=dict(
                                page_id=tf.train.Feature(int64_list=tf.train.Int64List(value=[page_id])),
                                word_ids=tf.train.Feature(int64_list=tf.train.Int64List(value=word_ids)),
                                entity_ids=tf.train.Feature(int64_list=tf.train.Int64List(value=entity_ids)),
                                entity_position_ids=tf.train.Feature(int64_list=Int64List(value=entity_position_ids)),
                            )
                        )
                    )
                    ret.append((example.SerializeToString()))

                words = []
                links = []
        return ret
