APP_NAME=EnglishLoenese
GO_MAIN=gui/main.go
PY_SRC=model/infer.py
MODEL=model/crnn_model.pth
ASSETS=assets/Loen.otf assets/icon.png
PYTHON_ENV=model/venv
PIP=$(PYTHON_ENV)/bin/pip

.PHONY: all clean build-go bundle venv

all: clean venv build-go bundle

venv:
	python3 -m venv venv
	venv/bin/pip install --upgrade pip
	venv/bin/pip install numpy pillow pyinstaller onnxruntime fastapi uvicorn python-multipart

build-go:
	CGO_ENABLED=1 go build -o $(APP_NAME) $(GO_MAIN)

bundle:
	mkdir -p $(APP_NAME).app/Contents/MacOS
	mkdir -p $(APP_NAME).app/Contents/Resources
	cp $(APP_NAME) $(APP_NAME).app/Contents/MacOS/
	cp model/crnn_model.pth $(APP_NAME).app/Contents/Resources/
	cp model/crnn_model.onnx $(APP_NAME).app/Contents/Resources/
	cp $(MODEL) $(APP_NAME).app/Contents/Resources/
	cp assets/Loen.otf $(APP_NAME).app/Contents/Resources/
	cp assets/icon.png $(APP_NAME).app/Contents/Resources/
	cp model/charset.json $(APP_NAME).app/Contents/Resources/
	cp Info.plist $(APP_NAME).app/Contents/
	cp model/server.py $(APP_NAME).app/Contents/Resources/
	rm $(APP_NAME)

clean:
	rm -rf $(GO_BIN) $(APP_NAME).app
	cd model && rm -rf build $(APP_NAME)
