import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
total_epoch = 10
max_len = 32
batch = 64
learning_rate = 0.001
DROPOUT = 0.2 # 0.2, 0.3, 0.4


embedding_size = 100
lstm_hidden_size = 200


data_path = "../../../Datasets/MultiWOZ_2.2"
