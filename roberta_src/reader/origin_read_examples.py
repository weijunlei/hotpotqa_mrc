import json
from tqdm import tqdm
from origin_reader_helper import is_whitespace, HotpotQAReaderExample


def reader_read_examples(input_file,
                         related_paragraph_file,
                         tokenizer,
                         is_training='train'):
    """ 读取数据 """
    data = json.load(open(input_file, 'r', encoding='utf-8'))
    # 测试结果
    # data = data[:100]
    related_paragraph_dict = json.load(open(related_paragraph_file, 'r'))
    examples = []
    fail_title_has_answer_num = 0
    fail_count = 0
    diff_num = 0

    for info in tqdm(data):
        context = ""
        question = info['question']
        if is_training == 'train':
            answer = info['answer']
            answer_label = info['labels'][0]
        elif is_training == 'dev':
            answer = info['answer']
            answer_label = None
        else:
            answer = None
            answer_label = None
        # 只获取最佳答案的情况，即只有一个
        if is_training == 'test':
            supporting_facts = []
        else:
            supporting_facts = info['supporting_facts']
        supporting_facts_dict = set(['{}${}'.format(x[0], x[1]) for x in supporting_facts])
        qas_id = info['_id']
        cur_length = len(context)
        # 初始化
        sentence_cls = []
        if is_training == 'train':
            sentence_labels = []
        start_position = None
        end_position = None
        if is_training == 'train':
            if answer.lower() == 'yes':
                start_position = -1
                end_position = -1
            if answer.lower() == 'no':
                start_position = -2
                end_position = -2
        full_sentence_mask = []
        full_sentence_labels = []
        char_to_matrix = []
        input_sentence_idx = 1  # 输入到模型中句子编号
        for context_idx, context_info in enumerate(info['context']):
            title, sentences = context_info
            # 判断句子是否在相关段落中
            get_related_p_set = set(related_paragraph_dict[qas_id])
            if context_idx in get_related_p_set:
                full_sentence_mask += [1 for sentence in sentences if sentence.strip() != '']
            else:
                full_sentence_mask += [0] * len(sentences)
            for sent_idx, sentence in enumerate(sentences):
                if context_idx in get_related_p_set and sentence.strip() == '':
                    continue
                if '{}${}'.format(title, sent_idx) in supporting_facts_dict:
                    full_sentence_labels.append(1)
                else:
                    full_sentence_labels.append(0)
            if is_training == 'train' and title == answer_label[0] and context_idx not in get_related_p_set:
                fail_title_has_answer_num += 1
                break
            if context_idx in get_related_p_set:
                offset = 0
                for sent_idx, sentence in enumerate(sentences):
                    if sentence.strip() == '':
                         continue
                    # 非第一句按照空格分隔不同句子
                    if context == '':
                        white_num = 0
                        # 去除掉开头的空格
                        while is_whitespace(sentence[0]):
                            white_num += 1
                            sentence = sentence[1:]
                        offset += white_num
                    elif not sentence.startswith(' '):
                        context += ' '
                        cur_length += 1
                        char_to_matrix += [input_sentence_idx]
                    context += sentence
                    char_to_matrix += [input_sentence_idx] * len(sentence)
                    input_sentence_idx += 1
                    sentence_cls.append(cur_length)
                    if is_training == 'train' and title == answer_label[0]:
                        if offset <= answer_label[1] < offset + len(sentence):
                            start_position = cur_length + answer_label[1] - offset
                        if offset < answer_label[2] <= offset + len(sentence):
                            end_position = cur_length + answer_label[2] - offset
                    cur_length += len(sentence)
                    offset += len(sentence)
                    if is_training == 'train':
                        if '{}${}'.format(title, sent_idx) in supporting_facts_dict:
                            sentence_labels.append(1)
                        else:
                            sentence_labels.append(0)
        # 测试集无答案
        if is_training == 'train' and start_position is None:
            fail_count += 1
            continue
        # 消除多个空格
        doc_tokens = []
        char2new_char = []
        prev_is_whitespace = True
        # TODO: check the -2 [CLS] yes no
        char_count = -2
        for ch in context:
            if is_whitespace(ch):
                prev_is_whitespace = True
                char2new_char.append(char_count + 1)
            else:
                if prev_is_whitespace:
                    doc_tokens.append(ch)
                    char_count += 2
                else:
                    # 非新词补到最后一个单词中去
                    doc_tokens[-1] += ch
                    char_count += 1
                prev_is_whitespace = False
                char2new_char.append(char_count)
        if prev_is_whitespace:
            char2new_char.append(char_count + 1)
        context = ' '.join(doc_tokens)
        new_sentence_cls = []
        newchar2matrix = [0] * len(context)
        for ctm_idx, ctm in enumerate(char_to_matrix):
            # TODO: check the bound
            if is_training == 'train' and char2new_char[ctm_idx] >= len(context):
                continue
            newchar2matrix[char_to_matrix[ctm_idx]] = ctm
        for sc in sentence_cls:
            new_sentence_cls.append(char2new_char[sc])
        if is_training == 'train':
            new_start_position = char2new_char[start_position]
            if end_position == len(char2new_char):
                new_end_position = char2new_char[end_position - 1] + 1
            else:
                new_end_position = char2new_char[end_position]
        char2word_offset = []
        subword2matrix = []
        doc_subwords = []
        subword2origin_index = []
        new_context_length = 0
        for doc_token_idx, doc_token in enumerate(doc_tokens):
            sub_tokens = tokenizer.tokenize(doc_token)
            tokens_length_sum = 0
            new_context_length += len(doc_token)
            unk_count = 0
            for sbt_idx, sub_token in enumerate(sub_tokens):
                token_len = len(sub_token)
                doc_subwords.append(sub_token)
                subword2origin_index.append(doc_token_idx)
                if len(char2word_offset) < len(newchar2matrix):
                    subword2matrix.append(newchar2matrix[len(char2word_offset)])
                else:
                    subword2matrix.append(newchar2matrix[len(char2word_offset) - 1])
                # TODO: check -2 原因
                if sub_token.startswith('##'):
                    token_len -= 2
                if sub_token == '[UNK]':
                    unk_count += 1
                    #TODO 这行什么意思？
                    if len(sub_tokens) == sbt_idx + 1:
                        token_len = len(doc_token) - tokens_length_sum
                    elif sub_tokens[sbt_idx + 1] == '[UNK]':
                        token_len = 1
                    else:
                        token_len = doc_token.find(sub_tokens[sbt_idx + 1][0], tokens_length_sum) - tokens_length_sum
                pre_length = len(char2word_offset)
                for new_idx in range(token_len):
                    if len(char2word_offset) < new_context_length:
                        char2word_offset.append(len(doc_subwords) - 1)
                tokens_length_sum += len(char2word_offset) - pre_length
            if doc_token_idx != len(doc_tokens) - 1:
                char2word_offset.append(len(doc_subwords))
                new_context_length += 1
            # TODO: check the function
            while len(char2word_offset) < new_context_length:
                char2word_offset.append(len(doc_subwords) - 1)
            while len(char2word_offset) > new_context_length:
                char2word_offset = char2word_offset[:-1]
            assert new_context_length == len(char2word_offset)
        assert len(char2word_offset) == len(context)
        if is_training == 'train':
            start_position_word = char2word_offset[new_start_position]
            if new_end_position == len(char2word_offset):
                end_position_word = char2word_offset[new_end_position - 1] + 1
            else:
                end_position_word = char2word_offset[new_end_position]
        # TODO: check the function
        sentence_word_cls = []
        for sc in new_sentence_cls:
            sentence_word_cls.append(char2word_offset[sc])
        sentence_cls_extend = []
        for swc_idx, swc in enumerate(sentence_word_cls):
            doc_subwords.insert(swc + swc_idx, '[UNK]')
            subword2origin_index.insert(swc + swc_idx, subword2origin_index[swc + swc_idx])
            subword2matrix.insert(swc + swc_idx, subword2matrix[swc + swc_idx])
            sentence_cls_extend.append(swc + swc_idx)
            if is_training == 'train':
                if start_position_word >= swc + swc_idx:
                    start_position_word += 1
                if end_position_word > swc + swc_idx:
                    end_position_word += 1
        if is_training == 'train':
            if answer.lower() == 'yes':
                start_position_word = -1
                end_position_word = -1
            if answer.lower() == 'no':
                start_position_word = -2
                end_position_word = -2
        if is_training != 'train':
            # 带标签的都无法获取
            start_position_word = None
            end_position_word = None
            sentence_labels = None
        example = HotpotQAReaderExample(
            qas_id=qas_id,
            question_text=question,
            origin_tokens=doc_tokens,
            doc_tokens=doc_subwords,
            subword2origin_index=subword2origin_index,
            origin_answer_text=answer,
            start_position=start_position_word,
            end_position=end_position_word,
            sentence_cls=sentence_cls_extend,
            sentence_labels=sentence_labels,
            full_sentences_mask=full_sentence_mask,
            full_sentences_labels=full_sentence_labels,
            subword2matrix=subword2matrix,
        )
        examples.append(example)
        new_answer = "".join(doc_subwords[start_position_word: end_position_word]).strip()
        origin_answer = answer.lower().replace(" ", "").strip()
        new_answer = new_answer.replace("#", "")
        if is_training == 'train' and new_answer != origin_answer and answer.lower() != 'yes' and answer.lower() != 'no':
            print("error in getting origin answer: {} with new answer: {}".format(origin_answer, new_answer))
            diff_num += 1
        # check result
    print("fail num: {}".format(fail_count))
    print("diff num: {}".format(diff_num))
    print("get example num: {}".format(len(examples)))
    return examples