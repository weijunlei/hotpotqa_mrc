import argparse
import spacy

nlp = spacy.load("en_core_web_sm")





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, help="hotpot origin input file",
                        default="../data/hotpot_data/hotpot_train_v1.1.json")
    parser.add_argument('--preprocessed_file', type=str, help="preprocessed file",
                        default="../data/hotpot_data/hotpot_train_preprocess_data_v3.json")
    config = parser.parse_args()
    max_overlap_preprocess(config=config)