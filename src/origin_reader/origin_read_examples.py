import json

from origin_reader_helper import SquadExample


def read_examples(input_file, filter_file, tokenizer, is_training):
    filter = json.load(open(filter_file, 'r'))
    examples = []
    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F or c=='\xa0':
            return True
        return False

    fail_count = 0
    lines = json.load(open(input_file, 'r', encoding='utf-8'))
    fail = 0
    for d in lines:
        context = ""
        question = d['question']
        answer = d['answer']
        answer_label = d['labels'][0]
        sup = d['supporting_facts']
        id = d['_id']
        length = len(context)
        sent_cls = []
        sent_lbs=[]
        start_position=None
        end_position=None
        if answer.lower()=='yes':
            start_position=-1
            end_position=-1
        if answer.lower()=='no':
            start_position=-2
            end_position=-2
        full_sents_mask=[]
        full_sents_lbs=[]
        char_to_matrix=[]
        sid=1
        for ind_con, con in enumerate(d['context']):#为了去掉句首的空白
            if ind_con in filter[id]:
                full_sents_mask+=[1 for con1 in con[1] if con1.strip()!='']
            else:
                full_sents_mask+=[0]*len(con[1])
            for indc1,c1 in enumerate(con[1]):
                if ind_con in filter[id] and c1.strip()=='':
                    continue
                if [con[0],indc1] in sup:
                    full_sents_lbs.append(1)
                else:
                    full_sents_lbs.append(0)
            if con[0]==answer_label[0] and ind_con not in filter[id]:
                fail+=1
                break
            if ind_con in filter[id]:
                offset = 0
                added = 0
                for inds, sent in enumerate(con[1]):
                    if sent.strip()=='':
                        continue
                    if context == '':
                        white1 = 0
                        while is_whitespace(sent[0]):
                            white1 += 1
                            sent = sent[1:]
                        offset+=white1
                    elif not sent.startswith(' '):
                        context+=' '
                        length+=1
                        char_to_matrix += [sid]
                    # context+='<unk>'+sent
                    context += sent
                    char_to_matrix += [sid]*len(sent)
                    sid += 1
                    sent_cls.append(length)
                    if con[0] == answer_label[0]:
                        if answer_label[1] >= offset and answer_label[1] < offset+len(sent):
                            start_position = length+answer_label[1]-offset#+5
                        # if answer_label[2]>=offset-white1 and answer_label[2]<=offset:
                        #     end_position=length
                        if answer_label[2] > offset and answer_label[2]<=offset+len(sent):
                            end_position = length+answer_label[2]-offset #+5
                    length += len(sent) # + 5
                    offset += len(sent)
                    if [con[0], inds] in sup:
                        sent_lbs.append(1)
                    else:
                        sent_lbs.append(0)
        if start_position is None:
            continue

        doc_tokens = []#为了消除多个whitespace连在一起
        char_to_newchar = []
        prev_is_whitespace = True
        char_count=-2
        for c in context:
            if is_whitespace(c):
                prev_is_whitespace = True
                char_to_newchar.append(char_count+1)
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                    char_count+=2
                else:
                    doc_tokens[-1] += c
                    char_count+=1
                prev_is_whitespace = False
                char_to_newchar.append(char_count)
        if prev_is_whitespace:
            char_to_newchar.append(char_count+1)

        context=' '.join(doc_tokens)
        sent_cls_n = []
        # print(id)
        newchar_to_matrix=[0]*len(context)
        for ind_ctm,ctm in enumerate(char_to_matrix):
            if char_to_newchar[ind_ctm]>=len(context):
                continue
            newchar_to_matrix[char_to_newchar[ind_ctm]]=ctm
        for sc in sent_cls:
            sent_cls_n.append(char_to_newchar[sc])
        start_position_n = char_to_newchar[start_position]
        if end_position == len(char_to_newchar):
            end_position_n = char_to_newchar[end_position - 1] + 1
        else:
            end_position_n = char_to_newchar[end_position]

        # an=context[start_position_n:end_position_n]
        # if an!=answer and answer!='yes' and answer!='no':
        #     print('error')
        char_to_word_offset = []
        subwords_to_matrix=[]
        doc_subwords=[]
        sub_to_orig_index=[]
        # cur_subword=0
        # cur_subword_offset=0
        conlen = 0
        for indt, dtoken in enumerate(doc_tokens):
            sub_tokens = tokenizer.tokenize(dtoken)
            sum_toklen = 0
            unkc = 0
            conlen += len(dtoken)
            for indst, subtoken in enumerate(sub_tokens):
                tok_len = len(subtoken)
                doc_subwords.append(subtoken)
                sub_to_orig_index.append(indt)
                if len(char_to_word_offset)<len(newchar_to_matrix):
                    subwords_to_matrix.append(newchar_to_matrix[len(char_to_word_offset)])
                else:
                    subwords_to_matrix.append(newchar_to_matrix[len(char_to_word_offset)-1])
                if subtoken.startswith('##'):
                    tok_len -= 2
                if subtoken == '[UNK]':
                    unkc += 1
                    if len(sub_tokens) == indst + 1:
                        tok_len = len(dtoken) - sum_toklen
                    elif sub_tokens[indst + 1] == '[UNK]':
                        tok_len = 1
                    else:
                        tok_len = dtoken.find(sub_tokens[indst + 1][0], sum_toklen) - sum_toklen
                prelen=len(char_to_word_offset)
                for rr in range(tok_len):
                    if len(char_to_word_offset) < conlen:
                        char_to_word_offset.append(len(doc_subwords) - 1)
                # while len(char_to_word_offset) > conlen:
                #     char_to_word_offset = char_to_word_offset[:-1]
                sum_toklen += len(char_to_word_offset)-prelen
            if indt != len(doc_tokens) - 1:
                char_to_word_offset.append(len(doc_subwords))
                conlen += 1
            while len(char_to_word_offset) < conlen:
                char_to_word_offset.append(len(doc_subwords) - 1)
            while len(char_to_word_offset) > conlen:
                char_to_word_offset = char_to_word_offset[:-1]
            assert conlen == len(char_to_word_offset)
        assert len(char_to_word_offset) == len(context)

        sent_cls_w = []
        for sc in sent_cls_n:
            sent_cls_w.append(char_to_word_offset[sc])
        start_position_w = char_to_word_offset[start_position_n]
        if end_position_n == len(char_to_word_offset):
            end_position_w = char_to_word_offset[end_position_n-1]+1
        else:
            end_position_w=char_to_word_offset[end_position_n]
        sent_cls_extend=[]
        for ind_scw, scw in enumerate(sent_cls_w):
            doc_subwords.insert(scw+ind_scw, '[UNK]')
            sub_to_orig_index.insert(scw+ind_scw, sub_to_orig_index[scw+ind_scw])
            subwords_to_matrix.insert(scw+ind_scw, subwords_to_matrix[scw+ind_scw])
            sent_cls_extend.append(scw+ind_scw)
            if start_position_w >= scw+ind_scw:
                start_position_w += 1
            if end_position_w > scw+ind_scw:
                end_position_w += 1

        if answer.lower() == 'yes':
            start_position_w = -1
            end_position_w = -1
        if answer.lower() == 'no':
            start_position_w = -2
            end_position_w = -2
        example = SquadExample(
            qas_id=id,
            question_text=question,
            orig_tokens=doc_tokens,
            sub_to_orig_index=sub_to_orig_index,
            doc_tokens=doc_subwords,
            orig_answer_text=answer,
            start_position=start_position_w,
            end_position=end_position_w,
            sent_cls=sent_cls_extend,
            sent_lbs=sent_lbs,
            full_sents_mask=full_sents_mask,
            full_sents_lbs=full_sents_lbs,
            subwords_to_matrix=subwords_to_matrix
        )
        examples.append(example)
    print('fail:',fail_count)
    # logging(input_file+' fail count '+str(fail_count))
    return examples


def read_dev_examples(input_file, filter_file,tokenizer,is_training):
    filter = json.load(open(filter_file, 'r'))
    examples = []
    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F or c=='\xa0':
            return True
        return False

    fail_count=0
    lines=json.load(open(input_file,'r',encoding='utf-8'))
    fail=0
    for d in lines:
        context = ""
        question = d['question']
        answer=d['answer']
        # answer_label=d['labels'][0]
        sup = d['supporting_facts']
        id=d['_id']
        length=len(context)
        sent_cls=[]
        full_sents_mask=[]
        full_sents_lbs=[]
        char_to_matrix = []
        sid = 1
        for ind_con,con in enumerate(d['context']):#为了去掉句首的空白
            if ind_con in filter[id]:
                full_sents_mask+=[1 for con1 in con[1] if con1.strip()!='']
            else:
                full_sents_mask+=[0]*len(con[1])
            for indc1,c1 in enumerate(con[1]):
                if ind_con in filter[id] and c1.strip()=='':
                    continue
                if [con[0],indc1] in sup:
                    full_sents_lbs.append(1)
                else:
                    full_sents_lbs.append(0)
            # if con[0]==answer_label[0] and ind_con not in filter[id]:
            #     fail+=1
            #     break
            if ind_con in filter[id]:
                offset=0
                for inds,sent in enumerate(con[1]):
                    if sent.strip()=='':
                        continue
                    if context=='':
                        white1=0
                        while is_whitespace(sent[0]):
                            white1+=1
                            sent=sent[1:]
                        offset+=white1
                    elif not sent.startswith(' '):
                        context+=' '
                        length+=1
                        char_to_matrix+=[sid]
                    # context+='<unk>'+sent
                    context +=sent
                    char_to_matrix += [sid] * len(sent)
                    sid += 1
                    sent_cls.append(length+0)
                    length += len(sent)# + 5
                    offset+=len(sent)
        # 'Ġ'
        # an0=context[start_position:end_position]
        # if start_position!=-1 and start_position!=-2:
        #     end_position=start_position+len(answer)

        doc_tokens = []#为了消除多个whitespace连在一起
        char_to_newchar = []
        prev_is_whitespace = True
        char_count=-2
        for c in context:
            if is_whitespace(c):
                prev_is_whitespace = True
                char_to_newchar.append(char_count+1)
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                    char_count+=2
                else:
                    doc_tokens[-1] += c
                    char_count+=1
                prev_is_whitespace = False
                char_to_newchar.append(char_count)
        if prev_is_whitespace:
            char_to_newchar.append(char_count+1)

        context=' '.join(doc_tokens)
        sent_cls_n = []
        newchar_to_matrix = [0] * len(context)
        for ind_ctm, ctm in enumerate(char_to_matrix):
            newchar_to_matrix[char_to_newchar[ind_ctm]] = ctm
        # print(id)
        for sc in sent_cls:
            sent_cls_n.append(char_to_newchar[sc])

        # an=context[start_position_n:end_position_n]
        # if an!=answer and answer!='yes' and answer!='no':
        #     print('error')
        char_to_word_offset = []
        subwords_to_matrix=[]
        doc_subwords=[]
        sub_to_orig_index=[]
        # cur_subword=0
        # cur_subword_offset=0
        conlen=0
        for indt, dtoken in enumerate(doc_tokens):
            sub_tokens = tokenizer.tokenize(dtoken)
            sum_toklen = 0
            unkc = 0
            conlen += len(dtoken)
            for indst, subtoken in enumerate(sub_tokens):
                tok_len = len(subtoken)
                doc_subwords.append(subtoken)
                sub_to_orig_index.append(indt)
                if len(char_to_word_offset)<len(newchar_to_matrix):
                    subwords_to_matrix.append(newchar_to_matrix[len(char_to_word_offset)])
                else:
                    subwords_to_matrix.append(newchar_to_matrix[len(char_to_word_offset)-1])
                if subtoken.startswith('##'):
                    tok_len -= 2
                if subtoken == '[UNK]':
                    unkc += 1
                    if len(sub_tokens) == indst + 1:
                        tok_len = len(dtoken) - sum_toklen
                    elif sub_tokens[indst + 1] == '[UNK]':
                        tok_len = 1
                    else:
                        tok_len = dtoken.find(sub_tokens[indst + 1][0], sum_toklen) - sum_toklen
                prelen=len(char_to_word_offset)
                for rr in range(tok_len):
                    if len(char_to_word_offset) < conlen:
                        char_to_word_offset.append(len(doc_subwords) - 1)
                # while len(char_to_word_offset) > conlen:
                #     char_to_word_offset = char_to_word_offset[:-1]
                sum_toklen += len(char_to_word_offset)-prelen
            if indt != len(doc_tokens) - 1:
                char_to_word_offset.append(len(doc_subwords))
                conlen+=1
            while len(char_to_word_offset) < conlen:
                char_to_word_offset.append(len(doc_subwords) - 1)
            while len(char_to_word_offset) > conlen:
                char_to_word_offset = char_to_word_offset[:-1]
            assert conlen == len(char_to_word_offset)
        assert len(char_to_word_offset) == len(context)
        assert len(char_to_word_offset) == len(context)
        # for indc,c in enumerate(context):
        #     char_to_word_offset.append(cur_subword)
        #     cur_subword_offset += len(c.encode('utf-8'))
        #     while cur_subword_offset>=len(doc_subwords[cur_subword]):
        #         cur_subword_offset-=len(doc_subwords[cur_subword])
        #         subwords_to_matrix.append(newchar_to_matrix[indc])
        #         cur_subword += 1
        #         if cur_subword>=len(doc_subwords):
        #             break
        sent_cls_w=[]
        for sc in sent_cls_n:
            sent_cls_w.append(char_to_word_offset[sc])
        sent_cls_extend=[]
        for ind_scw,scw in enumerate(sent_cls_w):
            doc_subwords.insert(scw+ind_scw,'[UNK]')
            sub_to_orig_index.insert(scw+ind_scw,sub_to_orig_index[scw+ind_scw])
            subwords_to_matrix.insert(scw+ind_scw,subwords_to_matrix[scw+ind_scw])
            sent_cls_extend.append(scw+ind_scw)
        if 0 in subwords_to_matrix:
            print('error')
        # actual_text = "".join(doc_subwords[start_position_w:end_position_w]).replace('Ġ',' ').strip()
        # cleaned_answer_text = " ".join(whitespace_tokenize(answer))
        # if actual_text!=cleaned_answer_text and answer!='yes':
        #     print(actual_text)
        #     print(cleaned_answer_text)
        #     print()

        example = SquadExample(
            qas_id=id,
            question_text=question,
            orig_tokens=doc_tokens,
            sub_to_orig_index=sub_to_orig_index,
            doc_tokens=doc_subwords,
            orig_answer_text=answer,
            sent_cls=sent_cls_extend,
            full_sents_mask=full_sents_mask,
            full_sents_lbs=full_sents_lbs,
            subwords_to_matrix=subwords_to_matrix
        )
        examples.append(example)
    print('fail:',fail_count)
    # logging(input_file+' fail count '+str(fail_count))
    return examples