import json
import random
from tqdm import tqdm


class HotpotQAExample(object):
    """ HotpotQA 实例 """
    def __init__(self,
                 qas_id,
                 question_tokens,
                 title_tokens,
                 context_tokens,
                 label):
        self.qas_id = qas_id
        self.question_tokens = question_tokens
        self.title_tokens = title_tokens
        self.context_tokens = context_tokens,
        self.label = label

    def __repr__(self):
        qa_info = "qas_id:{} question:{}".format(self.qas_id, self.question_tokens)
        if self.label:
            qa_info += " paragraph label: {}".format(self.label)
        return qa_info

    def __str__(self):
        return self.__repr__()


class HotpotInputFeature(object):
    """ HotpotQA 特征"""
    def __init__(self,
                 unique_id,
                 example_id,
                 tokens,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label=None):
        self.unique_id = unique_id
        self.example_id = example_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label = label


def read_hotpotqa_examples(input_file,
                           is_training: str='train',
                           not_related_sample_rate: float = 0.25,
                           tokenizer=None):
    """获取原始数据 """
    data = json.load(open(input_file, "r"))
    examples = []
    related_num = 0
    for info in tqdm(data, desc="Reading examples from data..."):
        context = info["context"]
        question = info["question"]
        if is_training == 'train':
            supporting_facts = info["supporting_facts"]
        else:
            supporting_facts = []
        supporting_facts_set = set(["{}${}".format(x[0], x[1]) for x in supporting_facts])
        for idx, paragraph in enumerate(context):
            title, sentences = paragraph
            related = False
            for sent_idx, sent in enumerate(sentences):
                if '{}${}'.format(title, sent_idx) in supporting_facts_set:
                    related = True
            # 控制采样比例
            if is_training == 'train' and not related and random.random() > not_related_sample_rate:
                continue
            if related:
                related_num += 1
            example = HotpotQAExample(
                qas_id="{}_{}".format(info['_id'], idx),
                question_tokens=question,
                title_tokens=title,
                context_tokens=sentences,
                label=related
            )
            examples.append(example)
    print("all example num: {} related num: {}".format(len(examples), related_num))
    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length, is_training):
    """ 将实例转化为特征 """
    features = []
    unique_id = 0
    for example_index, example in enumerate(tqdm(examples, desc="convert example to feature...")):
        query_tokens = tokenizer.tokenize(example.question_tokens)
        query_length = len(query_tokens)
        max_context_length = max_seq_length - query_length - 4
        all_tokens = ['<s>'] + query_tokens + ['</s>']
        title = example.title_tokens
        sentences = example.context_tokens[0]
        all_tokens += tokenizer.tokenize(title)
        all_tokens += ['</s>']
        for sentence in sentences:
            all_tokens += tokenizer.tokenize(sentence)
            all_tokens += ['</s>']
        all_tokens = all_tokens[:max_seq_length]
        tokens_length = len(all_tokens)
        segment_ids = [0] * query_length + [1] * (tokens_length - query_length)
        input_mask = [1] * tokens_length
        while len(all_tokens) < max_seq_length:
            all_tokens.append('<pad>')
            segment_ids.append(0)
            input_mask.append(0)
        input_ids = tokenizer.convert_tokens_to_ids(all_tokens)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(input_ids) == max_seq_length
        if is_training == 'train' or is_training == 'dev':
            label = example.label
        else:
            label = None
        feature = HotpotInputFeature(unique_id="{}_{}".format(example.qas_id, unique_id),
                                     example_id=example_index,
                                     tokens=all_tokens,
                                     input_ids=input_ids,
                                     input_mask=input_mask,
                                     segment_ids=segment_ids,
                                     label=label
                                     )
        features.append(feature)
    return features



