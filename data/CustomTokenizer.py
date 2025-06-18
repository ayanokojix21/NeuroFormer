from tokenizers import Tokenizer, models, trainers, normalizers, pre_tokenizers, decoders, processors

# Path of data from which data is taken to train tokenizer
lm_file = 'NeuroFormer/data/raw/Language Modelling/train.txt'
chat_file = 'NeuroFormer/data/raw/Chatbot Data/train.txt'
qa_file = 'NeuroFormer/data/raw/QuesAns Data/train.txt'
trans_file = 'NeuroFormer/data/raw/Translation Data/train.txt'

# Path where the trained tokenizer is saved
lm_path = 'NeuroFormer/tokenizers/lm/tokenizer.json'
chat_path = 'NeuroFormer/tokenizers/chat/tokenizer.json'
qa_path = 'NeuroFormer/tokenizers/qa/tokenizer.json'
trans_path = 'NeuroFormer/tokenizers/trans/tokenizer.json'

# Training LM Tokenizer
lm_tokenizer = Tokenizer(models.BPE())
lm_tokenizer.normalizer = normalizers.NFKC()
lm_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

lm_trainer = trainers.BpeTrainer(
    vocab_size=16000, # Setting Vocab Size of 16000 for LM Tokenizer
    min_frequency=3,
    special_tokens=["<pad>", "<s>", "</s>", "<unk>"]
)

lm_tokenizer.train([lm_file], trainer=lm_trainer)
lm_tokenizer.save(lm_path)
print('LM Tokenizer Saved Successfully')

# Training Chatbot Tokenizer
chat_tokenizer = Tokenizer(models.BPE())
chat_tokenizer.normalizer = normalizers.NFKC()
chat_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

chat_trainer = trainers.BpeTrainer(
    vocab_size=8000, # Setting Vocab Size of 8000 for Chatbot Tokenizer
    min_frequency=3,
    special_tokens=["<pad>", "<s>", "</s>", "<unk>", "<user>", "<assistant>"]
)

chat_tokenizer.train([chat_file], trainer=chat_trainer)
chat_tokenizer.save(chat_path)
print('Chatbot Tokenizer Saved Successfully')

# Training QA Tokenizer
qa_tokenizer = Tokenizer(models.WordPiece(unk_token="<unk>"))
qa_tokenizer.normalizer = normalizers.BertNormalizer()
qa_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
qa_tokenizer.decoder = decoders.WordPiece(prefix="##")

qa_trainer = trainers.WordPieceTrainer(
    vocab_size=8000, # Setting Vocab Size of 8000 for QA Tokenizer
    min_frequency=3,
    special_tokens=["<pad>", "<s>", "</s>", "<unk>", "<context>", "<question>", "<answer>"]
)

qa_tokenizer.train([qa_file], trainer=qa_trainer)
qa_tokenizer.post_processor = processors.TemplateProcessing(
    single="<s> $A </s>",
    pair="<s> $A </s> </s> $B </s>",
    special_tokens=[
        ("<s>", qa_tokenizer.token_to_id("<s>")),
        ("</s>", qa_tokenizer.token_to_id("</s>"))
    ]
)

qa_tokenizer.save(qa_path)
print('Ques Ans Tokenizer Saved Successfully')

# Training Translation Tokenizer 
trans_tokenizer = Tokenizer(models.BPE())
trans_tokenizer.normalizer = normalizers.BertNormalizer()
trans_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

trans_trainer = trainers.BpeTrainer(
    vocab_size=16000, # Setting Vocab Size of 16000 for Translation Tokenizer
    min_frequency=3,
    special_tokens=["<pad>", "<s>", "</s>", "<unk>", "<src>", "<tgt>"]
)

trans_tokenizer.train([trans_file], trainer=trans_trainer)
trans_tokenizer.save(trans_path)
print('Translation Tokenizer Saved Successfully')
