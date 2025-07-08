# English-Loense Translator

A cross-platform application for translating between the Loense language and English, and vice versa.

## Features

- **Bidirectional translation**: Loense ↔ English
- **Simple GUI** for ease of use
- **Cross-platform**: Works on Windows, macOS, and Linux

## Requirements

- **Operating System**: Windows, Linux, or macOS
- **Python** (recommended version: 3.8+)

### Required Python Packages

- numpy
- pillow
- pyinstaller
- onnxruntime
- fastapi
- uvicorn
- python-multipart

#### Installation Commands

**For Windows:**
```sh
pip install numpy pillow pyinstaller onnxruntime fastapi uvicorn python-multipart
```

**For macOS:**
```sh
python3 -m pip install numpy pillow pyinstaller onnxruntime fastapi uvicorn python-multipart
```

## How to Run

> **Note:** Your operating system may flag this software as potentially harmful due to the absence of a license. Proceed at your own discretion.

### Windows

- Open:  
  `build/exe/EnglishLoense.exe`

### macOS

- Open:  
  `build/exe/EnglishLoense.app`

## Building Executable Files Yourself

> **Important:** Executable files must be placed in the `build/exe` directory for correct operation.

### Windows

```sh
cd gui
```
```sh
CGO_ENABLED=1 GOOS=windows GOARCH=amd64 CC=x86_64-w64-mingw32-gcc go build -o EnglishLoenese.exe
```
```sh
mv EnglishLoenese.exe/ ../build/exe/
```

### macOS

```sh
make
```
```sh
mv EnglishLoenese.app/ build/exe/
```

## Notes

- Ensure all dependencies are installed before building or running the application.
- If you encounter warnings about the application's trustworthiness, this is due to the absence of a software license.