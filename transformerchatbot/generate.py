import torch
from torch.nn import functional as F
import os

from utils import get_tokenizer
from model import Transformer, ModelArgs
from lora import Lora

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = os.path.join(os.getcwd(), "model", "snapshot.pt")

model_args = ModelArgs()
model = Transformer(model_args)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
print("lora weight adding...")
lora = Lora(model)
lora.freeze_non_lora_params()
lora.print_model_parameters()
lora.enable_disable_lora(enabled=True)
total_params, trainable_params = lora.count_parameters()
print(f"Toplam parametre sayısı: {total_params}")
print(f"Eğitilebilir parametre sayısı: {trainable_params}")
model = lora.model

tokenizer = get_tokenizer()

def generate_text(model, text: str, stop_token=[32000], max_token: int = 100, temprature: float = 1.0):
    model.eval()
    idx = torch.tensor([tokenizer.encode(text)], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_token):
            logits, _ = model(idx)
            logits = logits[:, -1, :] / temprature
            props = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(props, num_samples=1)
            if idx_next in stop_token:
                break
            idx = torch.cat((idx, idx_next), dim=1)
            
    return tokenizer.decode(idx[0].tolist())



text = ""

while True:
    inpt = input("Human: ")
    text = text + "<user>" + inpt + "<bot>"
    response = generate_text(model, inpt)
    text =  text + response
    print(text)