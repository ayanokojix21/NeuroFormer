from transformers import PreTrainedTokenizerFast

def load_tokenizer(task):
    
    path_map = {
        "lm": "NeuroFormer/tokenizers/lm/tokenizer.json",
        "chat": "NeuroFormer/tokenizers/chat/tokenizer.json",
        "qa": "NeuroFormer/tokenizers/qa/tokenizer.json",
        "trans": "NeuroFormer/tokenizers/trans/tokenizer.json"
    }

    special_tokens = {
        "lm": {"pad_token": "<pad>", "unk_token": "<unk>", "bos_token": "<s>", "eos_token": "</s>"},
        "chat": {
            "pad_token": "<pad>", "unk_token": "<unk>", "bos_token": "<s>", "eos_token": "</s>",
            "additional_special_tokens": ["<user>", "<assistant>"]
        },
        "qa": {
            "pad_token": "<pad>", "unk_token": "<unk>", "bos_token": "<s>", "eos_token": "</s>",
            "additional_special_tokens": ["<context>", "<question>", "<answer>"]
        },
        "trans": {
            "pad_token": "<pad>", "unk_token": "<unk>", "bos_token": "<s>", "eos_token": "</s>",
            "additional_special_tokens": ["<source>", "<target>"]
        }
    }
    
    # Creatining a Wrapper of PreTrainedTokenizerFast on Custom Tokenizers
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=path_map[task])
    tokenizer.add_special_tokens(special_tokens[task])
    return tokenizer