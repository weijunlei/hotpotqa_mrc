import argparse
import json
import os

import spacy

nlp = spacy.load("en_core_web_sm")

def generate_entity_label(config=None):
    """"""
    datas = None
    with open(config.input_file, "r") as f:
        datas = json.load(f)
    for data_idx, data in enumerate(datas):
        qas_id = data['_id']
        question = data['question']
        answer = data['answer']
        data['entity_info'] = []
        for ind_con, con in enumerate(data['context']):
            sent_entity_labels = []
            sents = con[1]
            for sent in sents:
                doc = nlp(sent)
                for ent in doc.ents:
                    ent_text = ent.text
                    ent_label = ent.text
            print("hello")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, help="hotpot origin input file",
                        default="../../data/hotpot_data/hotpot_train_labeled_data_v3.json")
    parser.add_argument('--preprocessed_file', type=str, help="preprocessed file",
                        default="../../data/hotpot_data/hotpot_train_labeled_data_v3_with_entity_label.json")
    config = parser.parse_args()
    generate_entity_label(config=config)
