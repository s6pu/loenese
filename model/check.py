import torch
import numpy as np
from PIL import Image
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, img_h, n_channels, n_classes):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.rnn = nn.LSTM(128 * (img_h // 4), 256, bidirectional=True, num_layers=2, batch_first=True)
        self.fc = nn.Linear(512, n_classes)

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(b, w, c * h)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x

def preprocess_image(img_path):
    img = Image.open(img_path).convert('L')
    img = img.resize((128, 32))
    img = np.array(img, dtype=np.float32) / 255.0
    img = torch.tensor(img).unsqueeze(0).unsqueeze(0)
    return img

def ctc_greedy_decoder(output, charset):
    output = output.permute(1, 0, 2)
    pred = output.argmax(2).squeeze(1).cpu().numpy()
    prev = -1
    decoded = []
    for p in pred:
        if p != prev and p != 0:
            decoded.append(charset[p - 1])
        prev = p
    return ''.join(decoded)

if __name__ == '__main__':
    # Path to test image
    img_path = 'tmp.png'
    # Load model checkpoint
    checkpoint = torch.load('../model/crnn_model.pth', map_location='cpu')
    charset = checkpoint['charset']
    model = CRNN(img_h=32, n_channels=1, n_classes=len(charset) + 1)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    # Preprocess and predict
    img_tensor = preprocess_image(img_path)
    with torch.no_grad():
        outputs = model(img_tensor)
        outputs = outputs.log_softmax(2)
        pred_text = ctc_greedy_decoder(outputs, charset)
    print("Recognized text:", pred_text)
