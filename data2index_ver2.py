import json
import glob
import random

import config as cfg


intent_label = "intents"
slot_label = "slots0"

def make_woz_dict(file_path):
    words, intents, slots = set(), set(), set()
    with open(file_path, "r") as json_file:
        for dialogue_id, dialogue in json.load(json_file).items():
            for turn in dialogue:
                if len(turn[intent_label]) > 1:
                    continue
                if len(turn[intent_label]) == 0:
                    continue

                real_length = len(turn["words"])
                if real_length > cfg.max_len:
                    continue
                    
                words.update(turn["words"])
                intents.update(turn[intent_label])
                slots.update(turn[slot_label])

    word_dict = {'UNK': 0, 'PAD': 1}
    for i, item in enumerate(sorted(words)):
        word_dict[item] = i + 2
    intent_dict = {intent: idx for idx, intent in enumerate(sorted(intents))}
    slot_dict = {slot: idx for idx, slot in enumerate(sorted(slots))}
    
    return word_dict, intent_dict, slot_dict

def make_woz_index(file_path):
    dataset = []
    with open(file_path, "r") as json_file:
        for dialogue_id, dialogue in json.load(json_file).items():
            for turn in dialogue:
                if len(turn[intent_label]) > 1:
                    continue
                if len(turn[intent_label]) == 0:
                    continue

                real_length = len(turn["words"])
                if real_length > cfg.max_len:
                    continue
                    
                words = [word_dict[word] if word in word_dict else word_dict["UNK"] for word in turn["words"] + ["PAD"] * (cfg.max_len - real_length)]
                slots = [slot_dict[slot] for slot in turn[slot_label] + ["O"] * (cfg.max_len - real_length)]
                intent = intent_dict[turn[intent_label][0]] if len(turn[intent_label]) else intent_dict["general"]
                dataset.append([words, real_length, slots, intent])
    return dataset


word_dict, intent_dict, slot_dict = make_woz_dict(f"{cfg.data_path}/train.json")

train_data = make_woz_index(f"{cfg.data_path}/train.json")
test_data = make_woz_index(f"{cfg.data_path}/test.json")

index2slot_dict = {}
for key in slot_dict:
    index2slot_dict[slot_dict[key]] = key


# print('Number of training samples: ', len(train_data))
# print('Number of test samples: ', len(test_data))
# print('Number of words: ', len(word_dict))
# print('Number of intent labels: ', len(intent_dict))
# print('Number of slot labels', len(slot_dict))
# print("#" * 50)
# print(intent_dict)
