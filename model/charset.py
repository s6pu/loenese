import torch
import json

checkpoint = torch.load('crnn_model.pth', map_location='cpu')
charset = checkpoint['charset']
with open('charset.json', 'w', encoding='utf-8') as f:
    json.dump(charset, f, ensure_ascii=False)