import collections

from origin_reader_helper import InputFeatures, _check_is_max_context


def convert_examples_to_features(examples, tokenizer,graph, max_seq_length,
                                 doc_stride,  is_training):
    """Loads a data file into a list of `InputBatch`s."""
    # full_graph=json.load(open(graph,'r'))
    unique_id = 1000000000
    features = []
    for (example_index, example) in enumerate(examples):
        query_tokens = tokenizer.tokenize(example.question_text)
        all_doc_tokens = example.doc_tokens
        # graph=full_graph[example.qas_id]
        # The -5 accounts for '<s>','yes','no', </s> and </s>
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 5

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple("DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)
        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = ["[CLS]","yes","no"]
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = [0,0,0]
            matrix=[0,0,0]

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = split_token_index
                is_max_context = _check_is_max_context(doc_spans, doc_span_index,split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                matrix.append(example.subwords_to_matrix[split_token_index])
                segment_ids.append(0)
            content_len=len(tokens)
            tokens.append("[SEP]")
            segment_ids.append(0)
            matrix.append(0)

            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)
            matrix += [0] * len(query_tokens) + [-1]

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                matrix.append(-1)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(matrix)==max_seq_length

            # mask = []
            # last = -1
            # for mi in range(max_seq_length):
            #     if matrix[mi] == -1:
            #         continue
            #     if matrix[mi] == last:
            #         mask[-1][-1] += 1
            #         continue
            #     last = matrix[mi]
            #     cur_mask = []
            #     prev = False
            #     for mj in range(max_seq_length):
            #         if matrix[mi] == -1 or matrix[mj] == -1:
            #             prev = False
            #             continue
            #         if graph[matrix[mi]][matrix[mj]] == 1:
            #             if not prev:
            #                 cur_mask.append([mi, mj, 1])
            #             else:
            #                 cur_mask[-1][-1] += 1
            #             prev = True
            #         else:
            #             prev = False
            #     cur_mask.append(1)
            #     mask.append(cur_mask)
            new_mask = []
            # for ma in mask:
            #     for maa in ma:
            #         if isinstance(maa, list):
            #             new_mask.append(maa + [ma[-1]])
            # while len(new_mask) < max_seq_length:
            #     new_mask.append([0, 0, 0, 0])

            start_position_f=None
            end_position_f=None
            if is_training:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length
                if example.start_position==-1 and example.end_position==-1:
                    start_position_f=1
                    end_position_f=1
                elif example.start_position==-2 and example.end_position==-2:
                    start_position_f=2
                    end_position_f=2
                else:
                    if example.start_position>=doc_start and example.end_position<=doc_end:
                        start_position_f=example.start_position-doc_start+3
                        end_position_f=example.end_position-doc_start+2
                    else:
                        start_position_f=0
                        end_position_f=0
                sent_mask=[0]*max_seq_length
                sent_lbs=[0]*max_seq_length
                sent_weight=[0]*max_seq_length
                for ind_cls,orig_cls in enumerate(example.sent_cls):
                    if orig_cls>=doc_start and orig_cls<doc_end:
                        sent_mask[orig_cls-doc_start+3]=1
                        if example.sent_lbs[ind_cls]==1:
                            sent_lbs[orig_cls-doc_start+3]=1
                            sent_weight[orig_cls-doc_start+3]=1
                        else:
                            sent_weight[orig_cls - doc_start + 3] = 0.5

            features.append(
                InputFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    start_position=start_position_f,
                    end_position=end_position_f,
                    sent_mask=sent_mask,
                    sent_lbs=sent_lbs,
                    sent_weight=sent_weight,
                    mask=new_mask,
                    content_len=content_len
                    ))
            unique_id += 1
    return features


def convert_dev_examples_to_features(examples, tokenizer,graph, max_seq_length,
                                 doc_stride,  is_training):
    """Loads a data file into a list of `InputBatch`s."""
    # full_graph = json.load(open(graph, 'r'))
    unique_id = 1000000000
    features = []
    for (example_index, example) in enumerate(examples):
        query_tokens = tokenizer.tokenize(example.question_text)
        all_doc_tokens = example.doc_tokens
        # graph=full_graph[example.qas_id]
        # The -5 accounts for '<s>','yes','no', </s> and </s>
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 5

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple("DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)
        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = ["[CLS]","yes","no"]
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = [0,0,0]
            matrix=[0,0,0]

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = split_token_index
                is_max_context = _check_is_max_context(doc_spans, doc_span_index,split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(0)
                matrix.append(example.subwords_to_matrix[split_token_index])
            content_len=len(tokens)
            tokens.append("[SEP]")
            segment_ids.append(0)
            matrix.append(-1)

            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)
            matrix += [0] * len(query_tokens) + [-1]

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                matrix.append(-1)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(matrix)==max_seq_length
            sent_mask = [0] * max_seq_length
            doc_start = doc_span.start
            doc_end = doc_span.start + doc_span.length
            for ind_cls, orig_cls in enumerate(example.sent_cls):
                if orig_cls >= doc_start and orig_cls < doc_end:
                    sent_mask[orig_cls - doc_start + 3] = 1

            mask=[]
            # last=-1
            # for mi in range(max_seq_length):
            #     if matrix[mi]==-1:
            #         continue
            #     if matrix[mi]==last:
            #         mask[-1][-1]+=1
            #         continue
            #     last=matrix[mi]
            #     cur_mask=[]
            #     prev = False
            #     for mj in range(max_seq_length):
            #         if matrix[mi] == -1 or matrix[mj] == -1:
            #             prev=False
            #             continue
            #         if graph[matrix[mi]][matrix[mj]] == 1:
            #             if not prev:
            #                 cur_mask.append([mi, mj, 1])
            #             else:
            #                 cur_mask[-1][-1] += 1
            #             prev=True
            #         else:
            #             prev=False
            #     cur_mask.append(1)
            #     mask.append(cur_mask)
            new_mask=[]
            # for ma in mask:
            #     for maa in ma:
            #         if isinstance(maa,list):
            #             new_mask.append(maa+[ma[-1]])
            # while len(new_mask) < max_seq_length:
            #     new_mask.append([0,0,0,0])

            features.append(
                InputFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    sent_mask=sent_mask,
                    mask=new_mask,
                    content_len=content_len))
            unique_id += 1
    return features