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
                 doc_span_index,
                 tokens,
                 input_ids,
                 input_mask,
                 segment_ids,
                 cls_mask,
                 cls_label=None,
                 cls_weight=None,
                 is_related=None,
                 roll_back=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.cls_mask = cls_mask
        self.cls_label = cls_label
        self.cls_weight = cls_weight
        self.is_related = is_related
        self.roll_back = roll_back


def read_hotpotqa_examples(input_file,
                           is_training: str = 'train',
                           not_related_sample_rate: float = 0.25):
    """ 获取原始数据 """
    data = json.load(open(input_file, 'r'))
    # 测试流程通过情况
    # data = data[:100]
    examples = []
    related_num = 0
    not_related_num = 0

    for info in data:
        context = info['context']
        question = info['question']
        if is_training == 'test':
            supporting_facts = []
        else:
            supporting_facts = info['supporting_facts']
        supporting_facts_dict = set(['{}${}'.format(x[0], x[1]) for x in supporting_facts])
        for idx, paragraph in enumerate(context):
            labels = []
            title, sentences = paragraph
            related = False
            for sent_idx, sent in enumerate(sentences):
                if '{}${}'.format(title, sent_idx) in supporting_facts_dict:
                    labels.append(1)
                    related = True
                else:
                    labels.append(0)
            # 控制训练时的负样本采样比例
            if is_training == 'train' and not related and random.random() > not_related_sample_rate:
                continue
            if related:
                related_num += 1
            else:
                not_related_num += 1
            example = HotpotQAExample(
                qas_id='{}_{}'.format(info['_id'], idx),
                question_tokens=question,
                context_tokens=paragraph,
                sentences_label=labels,
                paragraph_label=related
            )
            examples.append(example)
    print("dataset type: {} related num:{} not related num: {} related / not: {} sample rate: {}".format(
        is_training,
        related_num,
        not_related_num,
        related_num / not_related_num,
        not_related_sample_rate
    ))
    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length, is_training):
    """ 将实例转化为特征 """
    features = []
    related_sent_num = 0
    not_related_sent_num = 0
    for example_index, example in enumerate(tqdm(examples)):
        query_tokens = tokenizer.tokenize(example.question_tokens)
        # special tokens ['CLS'] ['SEP'] ['SEP']
        max_context_length = max_seq_length - len(query_tokens) - 3
        cur_context_length = 0
        query_length = len(query_tokens) + 2
        unique_id = 0
        all_tokens = ['[CLS]'] + query_tokens + ['[SEP]']
        cls_mask = [1] + [0] * (len(all_tokens) - 1)
        if is_training == 'train' or is_training == 'dev':
            cls_label = [1 if example.paragraph_label else 0] + [0] * (len(all_tokens) - 1)
        else:
            cls_label = [0] * len(all_tokens)
        cls_weight = [1] + [0] * (len(all_tokens) - 1)
        sent_idx = 0
        pre_sent1_length = None
        pre_sent2_length = None

        while sent_idx < len(example.sentences_label):
            sentence = example.context_tokens[1][sent_idx]
            sent_label = example.sentences_label[sent_idx]
            sentence_tokens = tokenizer.tokenize(sentence)
            if sent_label:
                related_sent_num += 1
            else:
                not_related_sent_num += 1
            if len(sentence_tokens) + 1 > max_context_length:
                sentence_tokens = sentence_tokens[:max_context_length - 1]
            roll_back = 0
            if cur_context_length + len(sentence_tokens) + 1 > max_context_length:
                """ 超出长度往后延两句 """
                all_tokens += ['[SEP]']
                tmp_len = len(all_tokens)
                input_ids = tokenizer.convert_tokens_to_ids(all_tokens) + [0] * (max_seq_length - tmp_len)
                query_ids = [0] * query_length + [1] * (tmp_len - query_length) + [0] * (max_seq_length - tmp_len)
                input_mask = [1] * tmp_len + [0] * (max_seq_length - tmp_len)
                cls_mask += [1] + [0] * (max_seq_length - tmp_len)
                cls_label += [0] + [0] * (max_seq_length - tmp_len)
                cls_weight += [0] + [0] * (max_seq_length - tmp_len)
                if pre_sent2_length is not None:
                    if pre_sent2_length + pre_sent1_length + len(sentence_tokens) + 1 <= max_context_length:
                        roll_back = 2
                    elif pre_sent1_length + len(sentence_tokens) + 1 <= max_context_length:
                        roll_back = 1
                elif not pre_sent1_length and pre_sent1_length + len(sentence_tokens) + 1 <= max_context_length:
                    roll_back = 1
                sent_idx -= roll_back
                # 判断是否有支撑句，若无则新判别为非支撑段落
                real_related = int(bool(sum(cls_label) - cls_label[0]))
                if real_related != cls_label[0]:
                    cls_label[0] = real_related
                assert len(cls_mask) == max_seq_length
                assert len(cls_label) == max_seq_length
                assert len(cls_weight) == max_seq_length
                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(query_ids) == max_seq_length
                feature = HotpotInputFeatures(unique_id='{}_{}'.format(example.qas_id, unique_id),
                                              example_index=example_index,
                                              doc_span_index=unique_id,
                                              tokens=all_tokens,
                                              input_ids=input_ids,
                                              input_mask=input_mask,
                                              segment_ids=query_ids,
                                              cls_mask=cls_mask,
                                              cls_label=cls_label,
                                              cls_weight=cls_weight,
                                              is_related=real_related,
                                              roll_back=roll_back
                                              )
                features.append(feature)
                unique_id += 1
                # 还原到未添加context前
                cur_context_length = 0
                all_tokens = ['[CLS]'] + query_tokens + ['[SEP]']
                cls_mask = [1] + [0] * (len(all_tokens) - 1)
                cls_label = [1 if example.paragraph_label else 0] + [0] * (len(all_tokens) - 1)
                cls_weight = [1] + [0] * (len(all_tokens) - 1)
            else:
                all_tokens += ['[UNK]'] + sentence_tokens  # unk
                cls_mask += [1] + [0] * len(sentence_tokens)
                cls_label += [sent_label] + [0] * len(sentence_tokens)
                cls_weight += [1 if sent_label else 0.2] + [0] * len(sentence_tokens)
                cur_context_length += len(sentence_tokens) + 1
                sent_idx += 1
            pre_sent2_length = pre_sent1_length
            pre_sent1_length = len(sentence_tokens) + 1
        all_tokens += ['[SEP]']
        cls_mask += [1]
        cls_label += [0]
        cls_weight += [0]
        tmp_len = len(all_tokens)
        input_ids = tokenizer.convert_tokens_to_ids(all_tokens) + [0] * (max_seq_length - tmp_len)
        query_ids = [0] * query_length + [1] * (tmp_len - query_length) + [0] * (max_seq_length - tmp_len)
        input_mask = [1] * tmp_len + [0] * (max_seq_length - tmp_len)
        cls_mask += [0] * (max_seq_length - tmp_len)
        cls_label += [0] * (max_seq_length - tmp_len)
        cls_weight += [0] * (max_seq_length - tmp_len)
        real_related = int(bool(sum(cls_label) - cls_label[0]))
        if real_related != cls_label[0]:
            cls_label[0] = real_related
        assert len(cls_mask) == max_seq_length
        assert len(cls_label) == max_seq_length
        assert len(cls_weight) == max_seq_length
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(query_ids) == max_seq_length
        feature = HotpotInputFeatures(unique_id='{}_{}'.format(example.qas_id, unique_id),
                                      example_index=example_index,
                                      doc_span_index=unique_id,
                                      tokens=all_tokens,
                                      input_ids=input_ids,
                                      input_mask=input_mask,
                                      segment_ids=query_ids,
                                      cls_mask=cls_mask,
                                      cls_label=cls_label,
                                      cls_weight=cls_weight,
                                      is_related=real_related,
                                      roll_back=0
                                      )
        features.append(feature)
    print('get feature num:{} related sentences num: {} not related senteces num:{}'.format(len(features),
                                                                                            related_sent_num,
                                                                                            not_related_sent_num))
    return features