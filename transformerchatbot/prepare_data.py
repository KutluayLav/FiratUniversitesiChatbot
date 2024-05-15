from utils import get_tokenizer
import json
from typing import List


tokenizer = get_tokenizer()
print(tokenizer.vocab_size)

def preprocess_dialogues(data_path: str, tokenizer) -> List[int]:

    with open(data_path, "r") as f:
        data = json.load(f)
    
    dialogue_array = []

    for dialogue in data:
        dialogue_text = ""
        for data in dialogue["dialog"]:
            if data["sender"] == "user":
                formatted_sentence = f"<user>{data['text']}\n"
            elif data["sender"] == "bot":
                formatted_sentence = f"<bot>{data['text']}\n"
            else:
                raise ValueError(f"Geçersiz gönderici: {data['sender']}")
            
            dialogue_text = dialogue_text + formatted_sentence
        dialogue_text = dialogue_text + tokenizer.eos_token
        dialogue_array.extend(tokenizer.encode(dialogue_text))

    return dialogue_array
