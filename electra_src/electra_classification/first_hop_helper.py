import json
import random
from tqdm import tqdm


class HotpotQAExample(object):
    """ HotpotQA 实例解析"""
    def __init__(self,
                 qas_id,
                 question_tokens,
                 context_tokens,
                 sentences_label=None,
                 paragraph_label=None):
        self.qas_id = qas_id
        self.question_tokens = question_tokens
        self.context_tokens = context_tokens
        self.sentences_label = sentences_label
        self.paragraph_label = paragraph_label

    def __repr__(self):
        qa_info = "qas_id:{} question:{}".format(self.qas_id, self.question_tokens)
        if self.sentences_label:
            qa_info += " sentence label:{}".format(''.join([str(x) for x in self.sentences_label]))
        if self.paragraph_label:
            qa_info += " paragraph label: {}".format(self.paragraph_label)
        return qa_info

    def __str__(self):
        return self.__repr__()


class HotpotInputFeatures(object):
    """ HotpotQA input features to model """
    def __init__(self,
                 unique_id,
                 example_index,
                 tokens,
                 input_ids,
                 input_mask,
                 segment_ids,
                 is_related=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.is_related = is_related


def read_hotpotqa_examples(input_file,
                           is_training: str = 'train',
                           not_related_sample_rate: float = 1.0):
    """ 获取原始数据 """
    data = json.load(open(input_file, 'r'))
    # if is_training == 'dev':
    #     data = data[:1000]
    examples = []
    good_examples = []
    bad_examples = []

    for info in tqdm(data, desc="read examples..."):
        context = info['context']
        question = info['question']
        if is_training == 'test':
            support_facts = []
        else:
            support_facts = info['supporting_facts']
        support_facts_set = set(['{}_{}'.format(x[0], x[1]) for x in support_facts])
        for paragraph_idx, paragraph in enumerate(context):
            title, sentences = paragraph
            related = False
            for sentence_idx, sentence in enumerate(sentences):
                if '{}_{}'.format(title, sentence_idx) in support_facts_set:
                    related = True
            example = HotpotQAExample(
                qas_id='{}_{}'.format(info['_id'], paragraph_idx),
                question_tokens=question,
                context_tokens=paragraph,
                paragraph_label=1 if related else 0,
            )
            if is_training != 'train' or related:
                good_examples.append(example)
            else:
                bad_examples.append(example)
    related_num = len(good_examples)
    not_related_num = len(bad_examples)
    random.shuffle(bad_examples)
    bad_examples = bad_examples[:int(not_related_sample_rate * related_num)]
    examples = good_examples
    examples.extend(bad_examples)
    print("good examples: {} bad examples: {} sample bad examples: {}".format(related_num,
                                                                              not_related_num,
                                                                              len(bad_examples)))
    return examples


def convert_example_to_features(examples,
                                tokenizer,
                                max_seq_length,
                                is_training,
                                cls_token='[CLS]',
                                sep_token='[SEP]',
                                unk_token='[UNK]',
                                pad_token='[PAD]'):
    """ 将example 转化为预训练模型可处理的特征 """
    features = []
    related_num = 0
    not_related_num = 0
    unique_id = 100000000000
    shorten_paragraph = 0
    for example_idx, example in enumerate(tqdm(examples, desc="convert example to feature...")):
        query_tokens = tokenizer.tokenize(example.question_tokens)
        # cls + query + sep + context + sep
        max_context_length = max_seq_length - len(query_tokens) - 3
        title, sentences = example.context_tokens
        context_tokens = []
        context_tokens.extend(tokenizer.tokenize(title))
        for sentence_idx, sentence in enumerate(sentences):
            context_tokens.extend(tokenizer.tokenize(sentence))
        if len(context_tokens) > max_context_length:
            shorten_paragraph += 1
        context_tokens = context_tokens[:max_context_length]
        all_tokens = [cls_token] + query_tokens + [sep_token] + context_tokens + [sep_token]
        # +2 means add cls_token/sep_token, +1 means add sep_token
        segment_ids = [0] * (len(query_tokens) + 2) + [1] * (len(context_tokens) + 1)
        input_mask = [1] * len(all_tokens)
        is_related = 0
        while len(all_tokens) != max_seq_length:
            all_tokens.append(pad_token)
            input_mask.append(0)
            segment_ids.append(0)
        input_ids = tokenizer.convert_tokens_to_ids(all_tokens)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        if is_training == 'train' or is_training == 'dev':
            is_related = example.paragraph_label
        feature = HotpotInputFeatures(unique_id='{}_{}'.format(example.qas_id, unique_id),
                                      example_index=example_idx,
                                      tokens=all_tokens,
                                      input_ids=input_ids,
                                      input_mask=input_mask,
                                      segment_ids=segment_ids,
                                      is_related=is_related)
        unique_id += 1
        if is_related:
            related_num += 1
        else:
            not_related_num += 1
        features.append(feature)
    print('all feature num: {} related feature num: {} not related num: {}'.format(related_num + not_related_num,
                                                                                   related_num,
                                                                                   not_related_num))
    print("shorten paragraph num: {}".format(shorten_paragraph))
    return features
