train_file = 'data/olid-training-v1.0.tsv'
test_file = 'data/testset-levela.tsv'
test_answer = 'data/labels-levela.csv'

label_column = 'subtask_a'
data_column = 'tweet'

unk_token = "<UNK>"  # "[UNK]"
sep_token = "<SEP>"  # "[SEP]"
pad_token = "<PAD>"  # "[PAD]"

unk_token_id = 1
sep_token_id = 3
pad_token_id = 0

# dictionary_file = 'data/tokenizer-vocab.txt'
tokenizer_path = 'models/tokenizer.model'
model_file = 'models/best-model.pth'
