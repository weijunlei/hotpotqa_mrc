import os
import collections
import json


def prediction_evaluate_origin(args,
                               paragraph_results,
                               labels,
                               thread=0.5):
    """ 对预测进行评估 """
    p_recall = p_precision = sent_em = sent_acc = sent_recall = 0
    all_count = 0
    new_para_result = {}
    for k, v in paragraph_results.items():
        q_id, context_id = k.split('_')
        context_id = int(context_id)
        if q_id not in new_para_result:
            new_para_result[q_id] = [[0] * 10, [0] * 10]
        new_para_result[q_id][0][context_id] = v
    for k, v in labels.items():
        q_id, context_id = k.split('_')
        context_id = int(context_id)
        new_para_result[q_id][1][context_id] = v[0]
    for k, v in new_para_result.items():
        all_count += 1
        p11 = p10 = p01 = p00 = 0
        max_v = max(v[0])
        min_v = min(v[0])
        max_logit = -100
        max_result = False
        # TODO: check v format
        for idx, (paragraph_result, label) in enumerate(zip(v[0], v[1])):
            if paragraph_result > max_logit:
                max_logit = paragraph_result
                max_result = True if label == 1 else max_result
            # MinMax Scaling
            paragraph_result = (paragraph_result - min_v) / (max_v - min_v)
            paragraph_result = 1 if paragraph_result > thread else 0
            if paragraph_result == 1 and label == 1:
                p11 += 1
            elif paragraph_result == 1 and label == 0:
                p10 += 1
            elif paragraph_result == 0 and label == 1:
                p01 += 1
            elif paragraph_result == 0 and label == 0:
                p00 += 1
            else:
                # TODO: check the function
                raise NotImplemented
        if p11 + p01 != 0:
            p_recall += p11 / (p11 + p01)
        else:
            print("error in calculate paragraph recall!")
        if p11 + p10 != 0:
            p_precision += p11 / (p11 + p10)
        else:
            print("error in calculate paragraph precision!")
        if p11 == 2 and p10 == 0:
            sent_em += 1
        if p01 == 0:
            sent_recall += 1
        if max_result:
            sent_acc += 1
    return sent_acc / all_count, p_precision / all_count, sent_em / all_count, sent_recall / all_count


def prediction_evaluate(args,
                        paragraph_results,
                        labels,
                        thread=0.5,
                        step=0):
    """ 对预测进行评估 """
    recall = 0
    precision = 0
    f1 = 0
    em = 0.0
    for k, true_info in labels.items():
        predict_info = paragraph_results[k]
        true_num = 0
        for predict_num in predict_info:
            if predict_num in true_info:
                true_num += 1
        if true_num == len(true_info):
            em += 1.0
        recall += 1.0 * true_num / len(true_info)
        precision += 1.0 * true_num / len(predict_info)
    recall = 1.0 * recall / len(labels)
    precision = 1.0 * precision / len(paragraph_results)
    f1 = 2.0 * (precision * recall) / (recall + precision)
    em = 1.0 * em / len(labels)
    return recall, precision, f1, em


def write_predictions(args, all_examples, all_features, all_results, is_training='train', has_sentence_result=True,
                      step=0):
    """ 将预测结果写入json文件 """
    paragraph_results = {}
    labels = {}
    dev_data = json.load(open(args.dev_file))
    for info in dev_data:
        context = info['context']
        get_id = info['_id']
        if len(context) <= 2:
            paragraph_results[get_id] = list(range(len(context)))
        supporting_facts = info['supporting_facts']
        supporting_facts_dict = set(['{}${}'.format(x[0], x[1]) for x in supporting_facts])
        for idx, paragraph in enumerate(context):
            title, sentences = paragraph
            related = False
            for sent_idx, sent in enumerate(sentences):
                if '{}${}'.format(title, sent_idx) in supporting_facts_dict:
                    related = True
            if related:
                if get_id not in labels:
                    labels[get_id] = []
                labels[get_id].append(idx)
    dev_related_paragraph_dict = json.load(open("{}/{}".format(args.first_predict_result_path, args.dev_related_paragraph_file), "r"))
    for p_result in all_results:
        unique_id = p_result.unique_id
        qas_id = unique_id.split("_")[0]
        logit = p_result.logit
        min_num = min(logit)
        min_idx = logit.index(min_num)
        get_related_paragraphs = dev_related_paragraph_dict[qas_id]
        get_related_paragraphs = sorted(get_related_paragraphs)
        predict_result = []
        for idx, paragraph_num in enumerate(get_related_paragraphs):
            if idx == min_idx:
                continue
            predict_result.append(paragraph_num)
        paragraph_results[qas_id] = predict_result
    # return paragraph_results, sentence_results, labels
    if is_training == 'test':
        return 0, 0, 0, 0
    else:
        return prediction_evaluate(args,
                                   paragraph_results=paragraph_results,
                                   labels=labels,
                                   step=step)
