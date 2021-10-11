import json
from tqdm import tqdm

from first_hop_data_helper import HotpotQAExample
from first_hop_data_helper import HotpotInputFeatures

def read_second_hotpotqa_examples(input_file,
                                  best_paragraph_file,
                                  related_paragraph_file,
                                  new_context_file,
                                  is_training: str = 'train',
                                  not_related_sample_rate: float = 0.25):
    """ 获取原始数据 """
    data = json.load(open(input_file, 'r'))
    best_paragraph = json.load(open(best_paragraph_file, 'r'))
    related_paragraph = json.load(open(related_paragraph_file, 'r'))
    new_context = json.load(open(new_context_file, 'r'))
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
        qas_id = info['_id']
        question = new_context[qas_id]
        best_paragraph_idx = best_paragraph[qas_id]
        for idx, paragraph in enumerate(context):
            if idx == best_paragraph_idx:
                continue
            labels = []
            title, sentences = paragraph
            related = False
            for sent_idx, sent in enumerate(sentences):
                if '{}${}'.format(title, sent_idx) in supporting_facts_dict:
                    labels.append(1)
                    related = True
                else:
                    labels.append(0)
            # 去除非相关的paragraph
            if is_training == 'train' and not related and idx not in related_paragraph[qas_id]: # and random.random() > not_related_sample_rate:
                continue
            if related:
                related_num += 1
            else:
                not_related_num += 1
            example = HotpotQAExample(
                qas_id='{}_{}'.format(qas_id, idx),
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


def convert_examples_to_second_features(examples, tokenizer, max_seq_length, is_training):
    """ 将实例转化为特征 """
    features = []
    related_sent_num = 0
    not_related_sent_num = 0
    get_qas_id = {}
    for example_index, example in enumerate(tqdm(examples, desc="convert examples to features...")):
        query_tokens = tokenizer.tokenize(example.question_tokens)
        # 当query+第一段结果长度大于512时的处理方法
        if len(query_tokens) >= 400:
            query_tokens = query_tokens[:300]
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
            cls_label = [0] + [0] * (len(all_tokens) - 1)
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
                elif pre_sent1_length is not None and pre_sent1_length + len(sentence_tokens) + 1 <= max_context_length:
                    roll_back = 1
                sent_idx -= roll_back
                # 判断是否有支撑句，若无则新判别为非支撑段落
                real_related = int(bool(sum(cls_label) - cls_label[0]))
                if real_related != cls_label[0]:
                    cls_label[0] = real_related
                try:
                    assert len(cls_mask) == max_seq_length
                except Exception as e:
                    import pdb; pdb.set_trace()
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
                cls_mask += [1] + [0] * (len(sentence_tokens) + 0)
                cls_label += [sent_label] + [0] * (len(sentence_tokens) + 0)
                cls_weight += [1 if sent_label else 0.2] + [0] * (len(sentence_tokens) + 0)
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
        # 二次判别看是否删除掉了支撑句
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