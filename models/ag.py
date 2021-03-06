from typing import Iterator, List, Dict
from allennlp.data import Instance
from allennlp.data.fields import TextField, LabelField

from allennlp.data.dataset_readers import DatasetReader
from allennlp.common.file_utils import cached_path
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from models.base_reader import BaseReader

class AGNewsDatasetReader(BaseReader):
    """
    DatasetReader for Trec Question Classification, one sentence per line, like

         LOC:city Tell me what city the Kentucky Horse Park is near ?
    """
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None, embeddings='default', spl=True) -> None:
        super().__init__(lazy=False, embeddings=embeddings, token_indexers=token_indexers, spl=spl)

    def text_to_instance(self, tokens: List[Token], tags: List[str] = None) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"tokens": sentence_field}

        if tags:
            label_field = LabelField(label=tags)
            fields["label"] = label_field

        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path, encoding="ISO-8859-1") as f:
            for line in f:
                full_tags, sentence = line.strip().split(',', 1)
                tag = full_tags.replace('\"','')
                sentence = sentence.replace('\"','')
                sentence = self.tokenize(sentence)
                yield self.text_to_instance([Token(word) for word in sentence], tag)
