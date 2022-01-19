import json
from tqdm import tqdm
from multiprocessing.pool import Pool
import copy
import multiprocessing
from origin_reader_helper import HotpotQAExample, is_whitespace


def question_text_process(question, tokenizer):
    """ 将question转化为token"""
    question_tokens = []
    doc_tokens = []
    prev_is_whitespace = True
    for idx, q_ch in enumerate(question):
        if is_whitespace(q_ch):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(q_ch)
            else:
                doc_tokens[-1] += q_ch
            prev_is_whitespace = False
    for d_token in doc_tokens:
        sub_tokens = tokenizer.tokenize(d_token)
        for ind_st, sub_token in enumerate(sub_tokens):
            question_tokens.append(sub_token)
    return question_tokens


is_training_global = False
tokenizer_global = None
sp_dict_global = None
sentence_token_global = None
paragraph_token_global = '<p>'


def get_origin_context_index(all_doc_token_index, all_doc_token_to_add_mark, add_mark_to_doc_token, doc_token_lens,
                             doc_to_char_offset, is_end):
    add_mark_idx = all_doc_token_to_add_mark[all_doc_token_index]
    doc_idx = add_mark_to_doc_token[add_mark_idx]
    doc_token_len = doc_token_lens[doc_idx]
    ch_idx = doc_to_char_offset[doc_idx]
    if is_end:
        ch_idx += doc_token_len
    return ch_idx


def process_single_data(data):
    global is_training_global
    global tokenizer_global
    global sp_dict_global
    global sentence_token_global
    global paragraph_token_global
    no_answer_num = 0
    diff_num = 0
    max_context_length = 509
    over_context_num = 0

    is_yesno = False
    context = ""
    qas_id = data["_id"]
    question = data['question']
    answer = data.get('answer', '')
    sup = data.get('supporting_facts', [])
    question_tokens = question_text_process(question, tokenizer=tokenizer_global)
    full_sentence_labels = []
    # 获取context文本到token
    has_answer = False
    answer_text = ''
    if is_training_global:
        answer_text = data['answer']
        answer_label = data['labels'][0]
        sent_lbs = []
        if answer.lower() == 'yes':
            has_answer = True
            start_position = -1
            end_position = -1
        elif answer.lower() == 'no':
            has_answer = True
            start_position = -2
            end_position = -2
    context = ''
    sentence_mask = []
    sentence_labels = []
    sentence_indexs = []
    all_examples = []
    paragraph_indexs = []
    paragraph_start = 0
    start_position = None
    end_position = None

    for context_idx, context_info in enumerate(data['context']):
        title, sentences = context_info
        cur_length = 0
        first_add = False
        if context_idx not in sp_dict_global[qas_id]:
            continue
        if is_training_global and not has_answer and title == answer_label[0]:
            has_answer = True
            first_add = True
        for sent_idx, sentence in enumerate(sentences):
            if is_training_global and title == answer_label[0] and context_idx not in sp_dict_global[qas_id]:
                break
            if context_idx in sp_dict_global[qas_id] and sentence.strip() == '':
                continue
            if sentence.strip() == '':
                continue
            sentence_indexs.append(len(context))
            sentence_mask.append(1)
            if first_add and is_training_global and answer_label[1] >= cur_length:
                answer_label[1] += 1
                answer_label[2] += 1
            context += ' ' + sentence
            cur_length += len(sentence) + 1
            if is_training_global:
                if [title, sent_idx] in sup:
                    sentence_labels.append(1)
                else:
                    sentence_labels.append(0)
        # if first_add and is_training_global and paragraph_start > 0:
        #     answer_label[1] -= 1
        #     answer_label[2] -= 1
        paragraph_indexs.append(len(context))
        if not has_answer:
            paragraph_start += len(context)
    if is_training_global:
        # 保证答案被抽取到
        if has_answer:
            answer_title = answer_label[0]
            start_position = answer_label[1] + paragraph_start
            end_position = answer_label[2] + paragraph_start
            has_answer = True
    doc_tokens = []
    char_to_word_offset = []
    doc_to_char_offset = []
    prev_is_whitespace = True
    for ch_idx, ch in enumerate(context):
        if is_whitespace(ch):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_to_char_offset.append(ch_idx)
                doc_tokens.append(ch)
            else:
                doc_tokens[-1] += ch
            prev_is_whitespace = False
        char_to_word_offset.append(len(doc_tokens) - 1)
    doc_to_char_offset.append(len(context))
    char_to_word_offset.append(len(doc_tokens))
    if is_training_global and has_answer and start_position >= 0:
        start_position = char_to_word_offset[start_position]
        end_position = char_to_word_offset[end_position]
    add_mark_context = ''
    add_mark_doc_tokens = copy.deepcopy(doc_tokens)
    doc_token_lens = []
    for doc_token in doc_tokens:
        doc_token_lens.append(len(doc_token))
    add_mark_to_doc_token = []
    add_mark_num = 0
    # 添加句子标识
    sentence_mark_indexs = []
    for sentence_index in sentence_indexs:
        if sentence_index >= len(char_to_word_offset):
            sentence_index = len(doc_tokens) - 1
        else:
            sentence_index = char_to_word_offset[max(sentence_index, 0)]
        sentence_index += 1
        add_mark_doc_tokens.insert(sentence_index + add_mark_num, sentence_token_global)
        sentence_mark_indexs.append(sentence_index + add_mark_num)
        if is_training_global:
            if has_answer and start_position < 0:
                continue
            if has_answer and sentence_index <= start_position - add_mark_num:
                start_position += 1
            if has_answer and sentence_index <= end_position - add_mark_num:
                end_position += 1
        add_mark_num += 1
    # add mask token转化为doc_token
    add_sentence_num = 0
    add_sentence_mask = []
    add_sentence_labels = []
    for add_idx, add_mark_doc_token in enumerate(add_mark_doc_tokens):
        add_mark_to_doc_token.append(add_idx - add_sentence_num)
        if add_idx in sentence_mark_indexs:
            add_sentence_mask.append(1)
            if is_training_global:
                add_sentence_labels.append(sentence_labels[add_sentence_num])
            add_sentence_num += 1
        else:
            add_sentence_mask.append(0)
            add_sentence_labels.append(0)
    add_mark_to_doc_token.append(len(doc_tokens))
    all_doc_tokens = []
    all_sentence_masks = []
    all_sentence_labels = []
    all_doc_token_to_add_mark = []
    add_mark_to_all_doc_token = []
    for add_token_idx, add_token in enumerate(add_mark_doc_tokens):
        sub_tokens = tokenizer_global.tokenize(add_token)

        if add_sentence_mask[add_token_idx] == 1:
            assert len(sub_tokens) == 1, "error in mark"
            all_sentence_masks.extend([1 for _ in range(len(sub_tokens))])
            if is_training_global:
                if add_sentence_labels[add_token_idx] == 1:
                    assert len(sub_tokens) == 1, "error in mark"
                    all_sentence_labels.extend([1 for _ in range(len(sub_tokens))])
                else:
                    all_sentence_labels.extend([0 for _ in range(len(sub_tokens))])
        else:
            all_sentence_masks.extend([0 for _ in range(len(sub_tokens))])
            all_sentence_labels.extend([0 for _ in range(len(sub_tokens))])
        add_mark_to_all_doc_token.append(len(all_doc_tokens))
        for sub_idx, sub_token in enumerate(sub_tokens):
            all_doc_tokens.append(sub_token)
            all_doc_token_to_add_mark.append(add_token_idx)
    add_mark_to_all_doc_token.append(len(all_doc_tokens))
    all_doc_token_to_add_mark.append(len(add_mark_doc_token))

    def _same_rate(origin_text, get_text):
        all_num = len(origin_text)
        origin_dict = {}
        for ch in origin_text:
            if ch not in origin_dict:
                origin_dict[ch] = 0
            origin_dict[ch] += 1
        same_num = 0
        for ch in get_text:
            if ch in origin_dict and origin_dict.get(ch) > 0:
                same_num += 1
                origin_dict[ch] -= 1
        return 1.0 * same_num / all_num

    if is_training_global:
        if answer.lower() == 'yes':
            start_position = -1
            end_position = -1
            is_yesno = True
        elif answer.lower() == 'no':
            start_position = -2
            end_position = -2
            is_yesno = True
        elif has_answer:
            start_position = add_mark_to_all_doc_token[start_position]
            end_position = add_mark_to_all_doc_token[min(end_position + 1, len(add_mark_to_all_doc_token) - 1)] - 1
            end_position = max(start_position, end_position)
            # get_answer = all_doc_tokens[start_position: end_position]
            # get_answer = ''.join(get_answer)
            # get_answer = get_answer.lower().replace("▁", "").replace("'", "").replace('"', "").replace(" ", "")
            # truly_answer_text = answer_text.lower().replace("▁", "").replace("'", "").replace('"', "").replace(" ", "")
        elif not has_answer:
            start_position = -3
            end_position = -3
            no_answer_num += 1
    sub_to_orig_index = []
    for sub_token_idx, sub_token in enumerate(all_doc_tokens):
        add_mark_idx = all_doc_token_to_add_mark[sub_token_idx]
        doc_idx = add_mark_to_doc_token[add_mark_idx]
        ch_idx = doc_to_char_offset[doc_idx]
        sub_to_orig_index.append(ch_idx)
    sub_to_orig_index.append(len(context))
    if has_answer and not is_yesno:
        back_start = get_origin_context_index(all_doc_token_index=start_position,
                                              all_doc_token_to_add_mark=all_doc_token_to_add_mark,
                                              add_mark_to_doc_token=add_mark_to_doc_token,
                                              doc_token_lens=doc_token_lens,
                                              doc_to_char_offset=doc_to_char_offset,
                                              is_end=False
                                              )
        back_end = get_origin_context_index(all_doc_token_index=end_position,
                                            all_doc_token_to_add_mark=all_doc_token_to_add_mark,
                                            add_mark_to_doc_token=add_mark_to_doc_token,
                                            doc_token_lens=doc_token_lens,
                                            doc_to_char_offset=doc_to_char_offset,
                                            is_end=True
                                            )
        back_get_answer = context[back_start: back_end]
        back_get_answer = back_get_answer.lower().replace("▁", "").replace("'", "").replace('"', "").replace(" ", "")
        truly_answer_text = answer_text.lower().replace("▁", "").replace("'", "").replace('"', "").replace(" ", "")
        if len(truly_answer_text) != 0 and _same_rate(truly_answer_text, back_get_answer) <= 0.5:
            print("diff answer qas id: {} origin answer: {} new get answer: {}".format(qas_id, answer_text,
                                                                                       context[back_start: back_end]))
            diff_num += 1
    sub_to_orig_index.append(len(context))
    if len(question_tokens) + len(all_doc_tokens) > max_context_length:
        over_context_num += 1
    assert len(all_doc_tokens) == len(all_sentence_masks), "error in sentence mask"
    if is_training_global:
        assert len(all_doc_tokens) == len(all_sentence_labels), "error in sentence label"
    example = HotpotQAExample(
        qas_id=qas_id,
        question_tokens=question_tokens,
        sentence_masks=all_sentence_masks,
        doc_tokens=all_doc_tokens,
        context=context,
        origin_answer_text=answer_text,
        question_text=question,
        all_doc_token_to_add_mark=all_doc_token_to_add_mark,
        add_mark_to_doc_token=add_mark_to_doc_token,
        doc_token_lens=doc_token_lens,
        doc_to_char_offset=doc_to_char_offset,
        sub_to_orig_index=sub_to_orig_index,
        start_position=start_position,
        end_position=end_position,
        sentence_labels=all_sentence_labels,
    )
    all_examples.append(example)
    result = {}
    result['all_examples'] = all_examples
    result['no_answer_num'] = no_answer_num
    result['diff_num'] = diff_num
    result['over_context_num'] = over_context_num
    return result


def read_examples(input_file, supporting_para_file, tokenizer, is_training, sentence_token='<e>'):
    # 处理后的数据
    global is_training_global
    global tokenizer_global
    global sp_dict_global
    global sentence_token_global
    no_answer_num = 0
    diff_num = 0
    over_context_num = 0
    datas = json.load(open(input_file, 'r', encoding='utf-8'))
    # 支撑段落
    sp_dict = json.load(open(supporting_para_file, 'r'))
    # 增加squad 的支撑段落
    for data in datas:
        get_id = data['_id']
        if 'level' in data and data['level'] == 'squad':
            sp_dict[get_id] = [0, ]
    # 转换后的examples
    examples = []
    no_answer_examples = 0
    is_training_global = is_training
    tokenizer_global = tokenizer
    sp_dict_global = sp_dict
    sentence_token_global = sentence_token
    # 多进程处理
    # pool_size = max(1, multiprocessing.cpu_count() // 1)
    # pool = Pool(pool_size)
    # for result in tqdm(pool.imap(func=process_single_data, iterable=datas),
    #                                           total=len(datas),
    #                                           desc="process examples..."):
    #     examples.extend(result['all_examples'])
    #     no_answer_num += result['no_answer_num']
    #     diff_num += result['diff_num']
    #     over_context_num += result['over_context_num']
    # pool.close()
    # pool.join()
    # 单进程处理
    for data_idx, data in enumerate(tqdm(datas)):
        result = process_single_data(data=data)
        examples.extend(result['all_examples'])
        no_answer_num += result['no_answer_num']
        diff_num += result['diff_num']
        over_context_num += result['over_context_num']
    print("all_example: {} no answer num: {} diff num: {} over context num: {}".format(
        len(examples),
        no_answer_num,
        diff_num,
        over_context_num
    ))
    return examples
