import json
import collections
from origin_reader_helper import HotpotQAInputFeatures, _check_is_max_context


def convert_examples_to_features(examples,
                                 tokenizer,
                                 max_seq_length,
                                 doc_stride,
                                 is_training,
                                 graph_file=None
                                 ):
    """ 将实例类转化为模型可处理输入 """
    unique_id = 1000000000
    features = []
    import pdb; pdb.set_trace()
    if graph_file is not None:
        full_graph = json.load(open(graph_file, 'r'))
    for example_index, example in enumerate(examples):
        query_tokens = tokenizer.tokenize(example.question_text)
        all_doc_tokens = example.doc_tokens
        if graph_file is not None:
            example_graph = full_graph[example.qas_id]
        # -5 是因为输入结构为<CLS> 'yes' 'no' <SEP> <SEP> <SEP>
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 5
        _DocSpan = collections.namedtuple("DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        # 滑动窗口获取文档
        while start_offset < len(all_doc_tokens):
            tmp_length = len(all_doc_tokens) - start_offset
            if tmp_length > max_tokens_for_doc:
                tmp_length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=tmp_length))
            if start_offset + tmp_length == len(all_doc_tokens):
                break
            start_offset += min(tmp_length, doc_stride)
        for doc_span_idx, doc_span in enumerate(doc_spans):
            all_input_tokens = ["[CLS]", "yes", "no"]
            token2origin_map = {}
            token_is_max_context = {}
            segment_ids = [0, 0, 0]
            for new_idx in range(doc_span.length):
                split_token_index = doc_span.start + new_idx
                token2origin_map[len(all_input_tokens)] = split_token_index
                # TODO: anlysis the function
                is_max_context = _check_is_max_context(doc_spans, doc_span_idx, split_token_index)
                token_is_max_context[len(all_input_tokens)] = is_max_context
                # TODO: chekc the doc info
                all_input_tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(0)
            cur_context_length = len(all_input_tokens)
            all_input_tokens.append("[SEP]")
            segment_ids.append(0)
            # TODO: check 矩阵有何不同，之前是训练集是-1,验证集是0,应该都是-1？
            # query tokens 处理
            for token in query_tokens:
                all_input_tokens.append(token)
                segment_ids.append(1)
            all_input_tokens.append("[SEP]")
            segment_ids.append(1)
            input_ids = tokenizer.convert_tokens_to_ids(all_input_tokens)
            # 真正的输入才会被标记为1
            input_mask = [1] * len(input_ids)
            # 填充
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            final_start_position = None
            final_end_position = None
            if is_training == 'train':
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length
                if example.start_position == -1 and example.end_position == -1:
                    final_start_position = 1
                    final_end_position = 1
                elif example.start_position == -2 and example.end_position == -2:
                    final_start_position = 2
                    final_end_position = 2
                else:
                    if example.start_position >= doc_start and example.end_position <= doc_end:
                        # 开始输入有3个特殊字符
                        final_start_position = example.start_position - doc_start + 3
                        final_end_position = example.end_position - doc_start + 2
                    else:
                        final_start_position = 0
                        final_end_position = 0
                # 初始化
                sentence_mask = [0] * max_seq_length
                sentence_labels = [0] * max_seq_length
                sentence_weight = [0] * max_seq_length
                for cls_idx, origin_sentence_cls in enumerate(example.sentence_cls):
                    if doc_start <= origin_sentence_cls < doc_end:
                        sentence_mask[origin_sentence_cls - doc_start + 3] = 1
                        if example.sentence_labels[cls_idx] == 1:
                            sentence_labels[origin_sentence_cls - doc_start + 3] = 1
                            sentence_weight[origin_sentence_cls - doc_start + 3] = 1
                        else:
                            # TODO:验证不同权重的作用
                            sentence_weight[origin_sentence_cls - doc_start + 3] = 0.5
            elif is_training == 'dev':
                sentence_mask = [0] * max_seq_length
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length
                for cls_idx, origin_sentence_cls in enumerate(example.sentence_cls):
                    if doc_start <= origin_sentence_cls < doc_end:
                        sentence_mask[origin_sentence_cls - doc_start + 3] = 1
                sentence_labels = None
                sentence_weight = None

            if graph_file is None:
                final_graph_mask = None
            features.append(
                HotpotQAInputFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_idx,
                    tokens=all_input_tokens,
                    token2origin_map=token2origin_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    start_position=final_start_position,
                    end_position=final_end_position,
                    sentence_mask=sentence_mask,
                    sentence_labels=sentence_labels,
                    sentence_weight=sentence_weight,
                    graph_mask=None,
                    content_len=cur_context_length
                )
            )
            unique_id += 1
    import pdb; pdb.set_trace()
    return features
