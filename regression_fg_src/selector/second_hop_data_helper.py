import json
from tqdm import tqdm
import multiprocessing
import random
from multiprocessing import Pool


class HotpotQAExample(object):
    """ HotpotQA 实例解析"""

    def __init__(self,
                 qas_id,
                 question_tokens,
                 first_paragraph_tokens,
                 second_paragraph_tokens,
                 third_paragraph_tokens,
                 paragraphs_label=None):
        self.qas_id = qas_id
        self.question_tokens = question_tokens
        self.first_paragraph_tokens = first_paragraph_tokens
        self.second_paragraph_tokens = second_paragraph_tokens
        self.third_paragraph_tokens = third_paragraph_tokens
        self.paragraphs_label = paragraphs_label

    def __repr__(self):
        qa_info = "qas_id:{} question:{}".format(self.qas_id, self.question_tokens)
        if self.paragraphs_label:
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
                 pq_end_pos,
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
        self.pq_end_pos = pq_end_pos
        self.cls_label = cls_label
        self.cls_weight = cls_weight
        self.is_related = is_related
        self.roll_back = roll_back


def read_second_hotpotqa_examples(args,
                                  input_file,
                                  related_paragraph_file,
                                  is_training: str = 'train',
                                  not_related_sample_rate: float = 0.25):
    """ 获取原始数据 """
    data = json.load(open(input_file, 'r'))
    related_paragraph_dict = json.load(open(related_paragraph_file, 'r'))
    # 测试流程通过情况
    # data = data[:100]
    examples = []
    related_num = 0
    not_related_num = 0
    skip_num = 0

    for info in tqdm(data, desc="reading examples..."):
        context = info['context']
        if len(context) <= 2:
            skip_num += 1
            continue
        question = info['question']
        input_labels = info["labels"][0]
        label_title, _, _ = input_labels
        if is_training == 'test':
            supporting_facts = []
        else:
            supporting_facts = info['supporting_facts']
        supporting_facts_dict = set(['{}${}'.format(x[0], x[1]) for x in supporting_facts])
        qas_id = info['_id']
        related_paragraphs = related_paragraph_dict[qas_id]
        related_paragraphs = sorted(related_paragraphs)
        # TODO: check content test
        # question = new_context[qas_id]
        question = question
        paragraphs_label = []
        example = HotpotQAExample(
            qas_id=qas_id,
            question_tokens=question,
            first_paragraph_tokens=None,
            second_paragraph_tokens=None,
            third_paragraph_tokens=None,
            paragraphs_label=None,
        )
        for para_idx, related_paragraph in enumerate(related_paragraphs):
            paragraph = context[related_paragraph]
            has_sup = False
            title, sentences = paragraph
            for sent_idx, sent in enumerate(sentences):
                if '{}${}'.format(title, sent_idx) in supporting_facts_dict:
                    has_sup = True
            paragraph_context = ''
            if has_sup:
                related_num += 1
                paragraphs_label.append(1)
            else:
                not_related_num += 1
                paragraphs_label.append(0)
            for sent_idx, sent in enumerate(sentences):
                paragraph_context += sent
            if para_idx == 0:
                example.first_paragraph_tokens = paragraph_context
            elif para_idx == 1:
                example.second_paragraph_tokens = paragraph_context
            else:
                example.third_paragraph_tokens = paragraph_context
        assert len(paragraphs_label) == 3, "get unexpected paragraph label"
        example.paragraphs_label = paragraphs_label
        examples.append(example)
    print("dataset type: {} related num:{} not related num: {} related / not: {} sample rate: {}, skip num: {}".format(
        is_training,
        related_num,
        not_related_num,
        related_num / not_related_num,
        not_related_sample_rate,
        skip_num
    ))
    return examples


global_tokenizer = None
global_max_seq_length = None
global_is_training = None
global_cls_token = None
global_sep_token = None
global_unk_token = None
global_pad_token = None


def second_example_process(data):
    """ 将打个example转化为预训练模型可处理的特征 """
    example, example_index = data
    features = []
    related_sent_num = 0
    not_related_sent_num = 0
    shorten_num = 0
    global global_tokenizer
    global global_max_seq_length
    global global_is_training
    global global_cls_token
    global global_sep_token
    global global_unk_token
    global global_pad_token
    unique_id = 0
    query_tokens = global_tokenizer.tokenize(example.question_tokens)
    # 当query+第一段结果长度大于512时的处理方法
    if len(query_tokens) >= 64:
        query_tokens = query_tokens[:64]
    roll_back = 0
    # special tokens ['CLS'] ['SEP'] first ['SEP'] second ['SEP'] third ['SEP']
    max_context_length = global_max_seq_length - len(query_tokens) - 5
    each_max_length = max_context_length // 3
    first_tokens = global_tokenizer.tokenize(example.first_paragraph_tokens)
    second_tokens = global_tokenizer.tokenize(example.second_paragraph_tokens)
    third_tokens = global_tokenizer.tokenize(example.third_paragraph_tokens)
    should_shorten = (len(first_tokens) + len(second_tokens) + len(third_tokens)) > max_context_length
    if should_shorten:
        shorten_num += 1
        first_tokens = first_tokens[:each_max_length]
        second_tokens = second_tokens[:each_max_length]
        third_tokens = third_tokens[:each_max_length]
    all_tokens = [global_cls_token] + query_tokens
    query_length = len(all_tokens)

    cls_label = [0] * len(all_tokens)
    cls_mask = [0] * len(all_tokens) + [1]
    if example.paragraphs_label[0] == 1:
        cls_label.append(1)
        related_sent_num += 1
    else:
        cls_label.append(0)
        not_related_sent_num += 1
    all_tokens += ['<p>'] + first_tokens
    cls_label += [0] * len(first_tokens)
    cls_mask += [0] * len(first_tokens) + [1]
    if example.paragraphs_label[1] == 1:
        cls_label.append(1)
        related_sent_num += 1
    else:
        cls_label.append(0)
        not_related_sent_num += 1
    all_tokens += ['<p>'] + second_tokens
    cls_label += [0] * len(second_tokens)
    cls_mask += [0] * len(second_tokens) + [1]
    if example.paragraphs_label[2] == 1:
        cls_label.append(1)
        related_sent_num += 1
    else:
        cls_label.append(0)
        not_related_sent_num += 1
    all_tokens += ['<p>'] + third_tokens
    cls_label += [0] * len(third_tokens)
    cls_mask += [0] * len(third_tokens)
    all_tokens += [global_sep_token]
    cls_label.append(0)
    cls_mask.append(0)
    tmp_len = len(all_tokens)
    context_end_idx = len(all_tokens)
    pq_end_pos = [query_length, context_end_idx]
    input_mask = [1] * len(all_tokens) + [0] * (global_max_seq_length - tmp_len)
    query_ids = [0] * query_length + [1] * (tmp_len - query_length) + [0] * (global_max_seq_length - tmp_len)

    while len(all_tokens) < global_max_seq_length:
        all_tokens.append(global_pad_token)
        cls_label.append(0)
        cls_mask.append(0)
    input_ids = global_tokenizer.convert_tokens_to_ids(all_tokens)
    cls_weight = cls_mask
    assert len(cls_mask) == global_max_seq_length
    assert len(cls_label) == global_max_seq_length
    assert len(cls_weight) == global_max_seq_length
    assert len(input_ids) == global_max_seq_length
    assert len(input_mask) == global_max_seq_length
    assert len(query_ids) == global_max_seq_length
    feature = HotpotInputFeatures(unique_id='{}_{}'.format(example.qas_id, unique_id),
                                  example_index=example_index,
                                  doc_span_index=unique_id,
                                  tokens=all_tokens,
                                  input_ids=input_ids,
                                  input_mask=input_mask,
                                  segment_ids=query_ids,
                                  cls_mask=cls_mask,
                                  pq_end_pos=pq_end_pos,
                                  cls_label=cls_label,
                                  cls_weight=cls_weight,
                                  is_related=sum(example.paragraphs_label),
                                  roll_back=roll_back
                                  )
    features.append(feature)
    result = {}
    result["features"] = features
    result["related_sent_num"] = related_sent_num
    result["not_related_sent_num"] = not_related_sent_num
    result["shorten_num"] = shorten_num
    return result


def convert_examples_to_second_features(examples,
                                        tokenizer,
                                        max_seq_length,
                                        is_training,
                                        cls_token=None,
                                        sep_token=None,
                                        unk_token=None,
                                        pad_token=None
                                        ):
    """ 将实例转化为特征 """
    global global_tokenizer
    global global_max_seq_length
    global global_is_training
    global global_cls_token
    global global_sep_token
    global global_unk_token
    global global_pad_token
    global_tokenizer = tokenizer
    global_max_seq_length = max_seq_length
    global_is_training = is_training
    global_cls_token = cls_token
    global_sep_token = sep_token
    global_unk_token = unk_token
    global_pad_token = pad_token
    features = []
    related_sent_num = 0
    not_related_sent_num = 0
    shorten_num = 0
    get_qas_id = {}
    pool_size = max(1, multiprocessing.cpu_count() // 4)
    pool = Pool(pool_size)
    datas = [(example, example_index) for example_index, example in enumerate(examples)]
    for result in tqdm(pool.imap(func=second_example_process, iterable=datas),
                       total=len(datas),
                       desc="Convert examples to features..."):
        features.extend(result["features"])
        related_sent_num += result["related_sent_num"]
        not_related_sent_num += result["not_related_sent_num"]
        shorten_num += result["shorten_num"]
    # for data in tqdm(datas, desc="Convert examples to features..."):
    #     result = second_example_process(data)
    #     features.extend(result["features"])
    #     related_sent_num += result["related_sent_num"]
    #     not_related_sent_num += result["not_related_sent_num"]
    print(
        'get feature num:{} related sentences num: {} not related senteces num:{} shorten_num: {}'.format(len(features),
                                                                                                          related_sent_num,
                                                                                                          not_related_sent_num,
                                                                                                          shorten_num))
    pool.close()
    pool.join()
    return features
