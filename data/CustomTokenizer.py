from tokenizers import Tokenizer, normalizers, pre_tokenizers, processors, trainers
from tokenizers.models import BPE
from transformers import PreTrainedTokenizerFast

# Path of data from which data is taken to train tokenizer
lm_file = 'NeuroFormer/data/raw/Language Modelling/train.txt'
chat_file = 'NeuroFormer/data/raw/Chatbot Data/train.txt'

# Path where the trained tokenizer is saved
tokenizer_path = 'NeuroFormer/tokenizers/tokenizer'

# Training LM Tokenizer
tokenizer = Tokenizer(BPE(unk_token='<unk>'))
tokenizer.normalizer = normalizers.NFKC()
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

lm_trainer = trainers.BpeTrainer(
    vocab_size=8000, # Setting Vocab Size of 8000 for Tokenizer
    min_frequency=3,
    special_tokens=["<pad>", "<s>", "</s>", "<unk>", "<user>", "<assistant>"]
)

tokenizer.train([lm_file, chat_file], trainer=lm_trainer)
tokenizer.post_processor = processors.TemplateProcessing(
    single="<s> $A </s>",
    pair="<s> $A </s> <s> $B </s>",
    special_tokens=[
        ("<s>", tokenizer.token_to_id("<s>")),
        ("</s>", tokenizer.token_to_id("</s>")),
    ],
)

# Wrap with PreTrainedTokenizerFast
fast_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    bos_token="<s>",
    eos_token="</s>",
    unk_token="<unk>",
    pad_token="<pad>",
    additional_special_tokens=["<user>", "<assistant>"]
)

fast_tokenizer.save_pretrained(tokenizer_path)
print('Tokenizer Saved Successfully')