import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch.nn as nn

class OCRDataset(Dataset):
    """
    Dataset for OCR: loads images and corresponding labels.
    """
    def __init__(self, labels_file, charset):
        self.samples = []
        with open(labels_file, 'r') as f:
            for line in f:
                img_path, label = line.strip().split(' ', 1)
                self.samples.append((img_path, label))
        self.charset = charset
        self.char_to_idx = {c: i+1 for i, c in enumerate(charset)}  # 0 is reserved for CTC blank

    def encode_label(self, label):
        return [self.char_to_idx[c] for c in label if c in self.char_to_idx]

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(os.path.join('../dataset/images', img_path)).convert('L')
        img = img.resize((128, 32))
        img = np.array(img, dtype=np.float32) / 255.0
        img = torch.tensor(img).unsqueeze(0)
        label_encoded = torch.tensor(self.encode_label(label), dtype=torch.long)
        return img, label_encoded, len(label_encoded)

    def __len__(self):
        return len(self.samples)

def get_charset(labels_file):
    """
    Builds the character set from the labels file.
    """
    charset = set()
    with open(labels_file, 'r') as f:
        for line in f:
            _, label = line.strip().split(' ', 1)
            charset.update(label)
    return ''.join(sorted(charset))

def collate_fn(batch):
    """
    Custom collate function for DataLoader to handle variable-length labels.
    """
    imgs, labels, label_lens = zip(*batch)
    imgs = torch.stack(imgs)
    labels = torch.cat(labels)
    label_lens = torch.tensor(label_lens, dtype=torch.long)
    return imgs, labels, label_lens

class CRNN(nn.Module):
    """
    CRNN model: CNN → BiLSTM → Linear.
    """
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

def train():
    labels_file = '../dataset/images/labels.txt'
    charset = get_charset(labels_file)
    n_classes = len(charset) + 1  # +1 for CTC blank
    dataset = OCRDataset(labels_file, charset)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    device = torch.device('cpu')
    model = CRNN(img_h=32, n_channels=1, n_classes=n_classes).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for step, (imgs, labels, label_lens) in enumerate(dataloader, 1):
            print(f"Epoch {epoch+1}/{num_epochs} | Step: {step}/{len(dataloader)}")
            imgs = imgs.to(device)
            labels = labels.to(device)
            label_lens = label_lens.to(device)
            batch_size = imgs.size(0)
            outputs = model(imgs)
            outputs = outputs.log_softmax(2)
            input_lens = torch.full(size=(batch_size,), fill_value=outputs.size(1), dtype=torch.long).to(device)
            loss = criterion(outputs.permute(1, 0, 2), labels, input_lens, label_lens)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {total_loss/len(dataloader):.4f}")

        # Save model checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss / len(dataloader),
            'charset': charset
        }, 'model/crnn_model.pth')
        print("Model saved to model/crnn_model.pth")

if __name__ == '__main__':
    train()
