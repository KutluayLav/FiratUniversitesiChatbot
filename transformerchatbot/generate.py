import torch
from torch.nn import functional as F
import os

from utils import get_tokenizer
from model import Model, ModelArgs

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = os.path.join(os.getcwd(), "model", "snapshot.pt")

model_args = ModelArgs()
model = Model(model_args)

model.load_state_dict(torch.load(model_path))
model.to(device)

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

for i in range(5):
    inpt = input("Human: ")
    inpt = "<user>" + inpt + "<bot>"
    response = generate_text(model, inpt)
    text = response
    print("Bot: ",response)

