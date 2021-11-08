import json
import random
from tqdm import tqdm


class HotpotQAExample(object):
    """
    HotpotQA 实例解析
    """

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


def read_examples(input_file,
                  is_training,
                  not_related_sample_rate: float = 0.25):
    data = json.load(open(input_file, 'r'))
    examples = []
    related_num = 0
    not_related_num = 0
    for info in data:
        context = info['context']
        question = info['question']
        sup = info['supporting_facts']
        for idx, paragraph in enumerate(context):
            sentences = paragraph
            labels = []
            related = False
            for sent_idx, s in enumerate(sentences[1]):
                if [sentences[0], sent_idx] in sup:
                    labels.append(1)
                    related = True
                else:
                    labels.append(0)
            # 控制训练时的负样本采样比例
            if not related and random.random() > not_related_sample_rate:
                continue
            if related:
                related_num += 1
            else:
                not_related_num += 1
            example = HotpotQAExample(
                qas_id=info['_id'] + '_' + str(idx),
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


def read_dev_examples(input_file, is_training):
    data = json.load(open(input_file, 'r'))
    examples = []
    for info in data:
        context = info['context']
        question = info['question']
        sup = info['supporting_facts']
        for idx, paragraph in enumerate(context):
            sentences = paragraph
            labels = []
            related = False
            for sent_idx, s in enumerate(sentences[1]):
                if [sentences[0], sent_idx] in sup:
                    labels.append(1)
                    related = True
                else:
                    labels.append(0)
            example = HotpotQAExample(
                qas_id=info['_id'] + '_' + str(idx),
                question_tokens=question,
                context_tokens=paragraph,
                sentences_label=labels,
                paragraph_label=related
            )
            examples.append(example)
    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 sent_overlap, is_training):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    related_sent_num = 0
    not_related_sent_num = 0
    for example_index, example in tqdm(enumerate(examples)):
        query_tokens = tokenizer.tokenize(example.question_tokens)
        segment1_len = 2 + len(query_tokens)
        max_len = max_seq_length - len(query_tokens) - 4
        length = 0
        unique_id = 0
        tokens = ['<s>'] + query_tokens + ['</s>'] + ['</s>']
        cls_mask = [1] + [0] * (len(tokens) - 1)
        cls_label = [1 if example.paragraph_label else 0] + [0] * (len(tokens) - 1)
        cls_weight = [1] + [0] * (len(tokens) - 1)
        i = 0
        prev1 = None

        while i < len(example.sentences_label):
            sent = example.context_tokens[1][i]
            label = example.sentences_label[i]
            sent_tokens = tokenizer.tokenize(sent)
            if label:
                related_sent_num += 1
            else:
                not_related_sent_num += 1
            if len(sent_tokens) + 1 > max_len:
                sent_tokens = sent_tokens[:max_len - 1]
            roll_back = 0
            if length + len(sent_tokens) + 1 > max_len:
                tokens += ['</s>']
                cls_mask += [1]
                cls_label += [0]
                cls_weight += [0]
                valid_len = len(tokens)
                input_ids = tokenizer.convert_tokens_to_ids(tokens) + [1] * (max_seq_length - valid_len)
                segment_ids = [0] * segment1_len + [1] * (valid_len - segment1_len) + [0] * (max_seq_length - valid_len)
                input_mask = [1] * valid_len + [0] * (max_seq_length - valid_len)
                cls_mask += [0] * (max_seq_length - valid_len)
                cls_label += [0] * (max_seq_length - valid_len)
                cls_weight += [0] * (max_seq_length - valid_len)
                if prev2 is not None:
                    if prev2 + prev1 + len(sent_tokens) + 1 <= max_len:
                        roll_back = 2
                    elif prev1 + len(sent_tokens) + 1 <= max_len:
                        roll_back = 1
                elif prev1 is not None and prev1 + len(sent_tokens) + 1 <= max_len:
                    roll_back = 1
                i -= roll_back
                real_related = int(bool(sum(cls_label) - cls_label[0]))
                if real_related != cls_label[0]:
                    cls_label[0] = real_related
                assert len(cls_mask) == max_seq_length
                assert len(cls_label) == max_seq_length
                assert len(cls_weight) == max_seq_length
                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length
                features.append(
                    HotpotInputFeatures(
                        unique_id=example.qas_id + '_' + str(unique_id),
                        example_index=example_index,
                        doc_span_index=unique_id,
                        tokens=tokens,
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        cls_mask=cls_mask,
                        cls_label=cls_label,
                        cls_weight=cls_weight,
                        is_related=real_related,
                        roll_back=roll_back))
                length = 0
                unique_id += 1
                tokens = ['<s>'] + query_tokens + ['</s>'] + ['</s>']
                cls_mask = [1] + [0] * (len(tokens) - 1)
                cls_label = [1 if example.paragraph_label else 0] + [0] * (len(tokens) - 1)
                cls_weight = [1] + [0] * (len(tokens) - 1)
            else:
                tokens += ['<unk>'] + sent_tokens
                cls_mask += [1] + [0] * (len(sent_tokens) + 0)
                cls_label += [label] + [0] * (len(sent_tokens) + 0)
                cls_weight += [1 if label else 0.2] + [0] * (len(sent_tokens) + 0)
                length += len(sent_tokens) + 1
                i += 1
            prev2 = prev1
            prev1 = len(sent_tokens) + 1
        tokens += ['</s>']
        cls_mask += [1]
        cls_label += [0]
        cls_weight += [0]
        valid_len = len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens) + [1] * (max_seq_length - valid_len)
        segment_ids = [0] * segment1_len + [1] * (valid_len - segment1_len) + [0] * (max_seq_length - valid_len)
        input_mask = [1] * valid_len + [0] * (max_seq_length - valid_len)
        cls_mask += [0] * (max_seq_length - valid_len)
        cls_label += [0] * (max_seq_length - valid_len)
        cls_weight += [0] * (max_seq_length - valid_len)
        real_related = int(bool(sum(cls_label) - cls_label[0]))
        if real_related != cls_label[0]:
            cls_label[0] = real_related
        assert len(cls_mask) == max_seq_length
        assert len(cls_label) == max_seq_length
        assert len(cls_weight) == max_seq_length
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        features.append(
            HotpotInputFeatures(
                unique_id=example.qas_id + '_' + str(unique_id),
                example_index=example_index,
                doc_span_index=unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                cls_mask=cls_mask,
                cls_label=cls_label,
                cls_weight=cls_weight,
                is_related=real_related,
                roll_back=0))
    print(len(features))
    print(related_sent_num, not_related_sent_num)
    return features
