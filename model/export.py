import torch
from server import CRNN  # импортируем вашу модель

def export_to_onnx():
    # Загружаем модель как в оригинальном коде
    checkpoint = torch.load('crnn_model.pth', map_location='cpu')
    charset = checkpoint['charset']
    model = CRNN(img_h=32, n_channels=1, n_classes=len(charset) + 1)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Создаем dummy input для трассировки
    dummy_input = torch.randn(1, 1, 32, 128)  # batch, channels, height, width
    
    # Экспортируем модель в ONNX
    torch.onnx.export(
        model,
        dummy_input,
        "crnn_model.onnx",
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 3: 'width'},  # динамические оси
            'output': {0: 'batch_size', 1: 'width'}
        }
    )

if __name__ == '__main__':
    export_to_onnx()