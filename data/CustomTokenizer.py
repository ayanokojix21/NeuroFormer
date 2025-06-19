from tokenizers import Tokenizer, models, trainers, normalizers, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

# Path of data from which data is taken to train tokenizer
lm_file = 'NeuroFormer/data/raw/Language Modelling/train.txt'
chat_file = 'NeuroFormer/data/raw/Chatbot Data/train.txt'

# Path where the trained tokenizer is saved
lm_path = 'NeuroFormer/tokenizers/lm/'
chat_path = 'NeuroFormer/tokenizers/chat/'

# Training LM Tokenizer
lm_tokenizer = Tokenizer(models.BPE(unk_token='<unk>'))
lm_tokenizer.normalizer = normalizers.NFKC()
lm_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

lm_trainer = trainers.BpeTrainer(
    vocab_size=16000, # Setting Vocab Size of 16000 for LM Tokenizer
    min_frequency=3,
    special_tokens=["<pad>", "<s>", "</s>", "<unk>"]
)

lm_tokenizer.train([lm_file], trainer=lm_trainer)
lm_tokenizer.post_processor = processors.TemplateProcessing(
    single="<s> $A </s>",
    pair="<s> $A </s> <s> $B </s>",
    special_tokens=[
        ("<s>", lm_tokenizer.token_to_id("<s>")),
        ("</s>", lm_tokenizer.token_to_id("</s>")),
    ],
)

# Wrap with PreTrainedTokenizerFast
lm_fast_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=lm_tokenizer,
    bos_token="<s>",
    eos_token="</s>",
    unk_token="<unk>",
    pad_token="<pad>",
)

lm_fast_tokenizer.save_pretrained(lm_path)
print('LM Tokenizer Saved Successfully')

# Training Chatbot Tokenizer
chat_tokenizer = Tokenizer(models.BPE(unk_token='<unk>'))
chat_tokenizer.normalizer = normalizers.NFKC()
chat_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

chat_trainer = trainers.BpeTrainer(
    vocab_size=8000, # Setting Vocab Size of 8000 for Chatbot Tokenizer
    min_frequency=3,
    special_tokens=["<pad>", "<s>", "</s>", "<unk>", "<user>", "<assistant>"]
)

chat_tokenizer.train([chat_file], trainer=chat_trainer)
chat_tokenizer.post_processor = processors.TemplateProcessing(
    single="<s> $A </s>",
    pair="<s> $A </s> <s> $B </s>",
    special_tokens=[
        ("<s>", chat_tokenizer.token_to_id("<s>")),
        ("</s>", chat_tokenizer.token_to_id("</s>")),
    ],
)

# Wrap with PreTrainedTokenizerFast
chat_fast_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=chat_tokenizer,
    bos_token="<s>",
    eos_token="</s>",
    unk_token="<unk>",
    pad_token="<pad>",
    additional_special_tokens=["<user>", "<assistant>"]
)

chat_fast_tokenizer.save_pretrained(chat_path)
print('Chatbot Tokenizer Saved Successfully')