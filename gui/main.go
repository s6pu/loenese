package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"io"
	"io/ioutil"
	"log"
	"mime/multipart"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/app"
	"fyne.io/fyne/v2/canvas"
	"fyne.io/fyne/v2/container"
	"fyne.io/fyne/v2/dialog"
	"fyne.io/fyne/v2/storage"
	"fyne.io/fyne/v2/widget"

	"golang.org/x/image/font"
	"golang.org/x/image/font/opentype"
	"golang.org/x/image/math/fixed"

	clipboard "github.com/skanehira/clipboard-image/v2"
)

const (
	IMG_HEIGHT = 64
	FONT_SIZE  = 48.0
	TMP_FILE   = "tmp.png"
)

func resourcePath(relPath string) string {
	devPath := filepath.Join("../assets", relPath)
	if _, err := os.Stat(devPath); err == nil {
		return devPath
	}

	exePath, err := os.Executable()
	if err != nil {
		return relPath
	}
	base := filepath.Dir(exePath)
	return filepath.Join(base, "..", "Resources", relPath)
}

func imageToText(imgPath string, port int) (string, error) {
	url := fmt.Sprintf("http://127.0.0.1:%d/predict/", port)

	file, err := os.Open(imgPath)
	if err != nil {
		return "", err
	}
	defer file.Close()

	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)
	part, err := writer.CreateFormFile("file", imgPath)
	if err != nil {
		return "", err
	}
	_, err = io.Copy(part, file)
	if err != nil {
		return "", err
	}
	writer.Close()

	req, err := http.NewRequest("POST", url, &buf)
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", writer.FormDataContentType())

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}

	type Response struct {
		Result string `json:"result"`
	}
	var res Response
	err = json.Unmarshal(body, &res)
	if err != nil {
		return "", err
	}

	return strings.TrimSpace(res.Result), nil
}

func saveClipboardImage(tmpFile string) error {
	r, err := clipboard.Read()
	if err != nil {
		return err
	}
	defer func() {
		if closer, ok := r.(io.Closer); ok {
			closer.Close()
		}
	}()
	f, err := os.Create(tmpFile)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = io.Copy(f, r)
	return err
}

func renderTextToImage(text string, fontPath string, fontSize float64, imgHeight int) (image.Image, int, error) {
	fontBytes, err := ioutil.ReadFile(fontPath)
	if err != nil {
		return nil, 0, err
	}
	fnt, err := opentype.Parse(fontBytes)
	if err != nil {
		return nil, 0, err
	}
	face, err := opentype.NewFace(fnt, &opentype.FaceOptions{
		Size:    fontSize,
		DPI:     72,
		Hinting: font.HintingNone,
	})
	if err != nil {
		return nil, 0, err
	}
	defer face.Close()

	dr := &font.Drawer{Face: face}
	textWidth := dr.MeasureString(text).Ceil() + 20
	img := image.NewRGBA(image.Rect(0, 0, textWidth, imgHeight))
	draw.Draw(img, img.Bounds(), &image.Uniform{color.White}, image.Point{}, draw.Src)

	dr = &font.Drawer{
		Dst:  img,
		Src:  image.NewUniform(color.Black),
		Face: face,
		Dot:  fixed.P(10, int(fontSize)+8),
	}
	dr.DrawString(text)
	return img, textWidth, nil
}

func concatImagesVertically(images []image.Image) (image.Image, int, int) {
	if len(images) == 0 {
		return nil, 0, 0
	}
	maxWidth := 0
	totalHeight := 0
	for _, img := range images {
		b := img.Bounds()
		if b.Dx() > maxWidth {
			maxWidth = b.Dx()
		}
		totalHeight += b.Dy()
	}
	out := image.NewRGBA(image.Rect(0, 0, maxWidth, totalHeight))
	draw.Draw(out, out.Bounds(), &image.Uniform{color.White}, image.Point{}, draw.Src)
	y := 0
	for _, img := range images {
		b := img.Bounds()
		draw.Draw(out, image.Rect(0, y, b.Dx(), y+b.Dy()), img, image.Point{}, draw.Over)
		y += b.Dy()
	}
	return out, maxWidth, totalHeight
}

func splitImageByLine(img image.Image, lineHeight int) []image.Image {
	b := img.Bounds()
	var images []image.Image
	for y := 0; y < b.Dy(); y += lineHeight {
		h := lineHeight
		if y+h > b.Dy() {
			h = b.Dy() - y
		}
		subImg := image.NewRGBA(image.Rect(0, 0, b.Dx(), h))
		draw.Draw(subImg, subImg.Bounds(), &image.Uniform{color.White}, image.Point{}, draw.Src)
		draw.Draw(subImg, subImg.Bounds(), img, image.Point{0, y}, draw.Over)
		images = append(images, subImg)
	}
	return images
}

func ocrLinesParallel(lineImages []image.Image, port int) ([]string, error) {
	var wg sync.WaitGroup
	results := make([]string, len(lineImages))
	errs := make([]error, len(lineImages))

	for i, img := range lineImages {
		wg.Add(1)
		go func(idx int, lineImg image.Image) {
			defer wg.Done()
			buf := new(bytes.Buffer)
			png.Encode(buf, lineImg)
			tmpFile := fmt.Sprintf("line_tmp_%d.png", idx)
			ioutil.WriteFile(tmpFile, buf.Bytes(), 0644)
			txt, err := imageToText(tmpFile, port)
			results[idx] = txt
			errs[idx] = err
			os.Remove(tmpFile)
		}(i, img)
	}
	wg.Wait()

	for _, err := range errs {
		if err != nil {
			return nil, err
		}
	}
	return results, nil
}

func getFreePort() (int, error) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, err
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port, nil
}

func main() {
	port, er := getFreePort()
	if er != nil {
		log.Fatalf("Failed to find free port: %v", er)
	}

	serverPath := resourcePath("server.py")
	cmd := exec.Command("uvicorn", "server:app", "--host", "0.0.0.0", "--port", fmt.Sprintf("%d", port))
	cmd.Dir = filepath.Dir(serverPath)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	err := cmd.Start()
	if err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}

	time.Sleep(2 * time.Second)

	myApp := app.New()

	iconPath := resourcePath("icon.png")
	icon, err := fyne.LoadResourceFromPath(iconPath)
	if err == nil {
		myApp.SetIcon(icon)
	}

	myWindow := myApp.NewWindow("English-Loenese")
	myWindow.Resize(fyne.NewSize(1000, 600))

	input := widget.NewMultiLineEntry()
	input.Wrapping = fyne.TextWrapWord
	input.SetPlaceHolder("Write text in English to translate to Loenese...")
	input.SetMinRowsVisible(10)

	imgResource := canvas.NewImageFromImage(nil)
	imgResource.FillMode = canvas.ImageFillContain
	imgViewWidth := 700
	imgViewHeight := 350

	imgBox := container.NewWithoutLayout(imgResource)
	imgBox.Resize(fyne.NewSize(float32(imgViewWidth), float32(imgViewHeight)))

	fontPath := resourcePath("Loen.otf")

	updateImage := func(text string) {
		lines := strings.Split(text, "\n")
		var imgs []image.Image
		maxW := 0
		for _, line := range lines {
			if line == "" {
				line = " "
			}
			img, w, err := renderTextToImage(line, fontPath, FONT_SIZE, IMG_HEIGHT)
			if err != nil {
				log.Println("Render error:", err)
				return
			}
			imgs = append(imgs, img)
			if w > maxW {
				maxW = w
			}
		}
		outImg, outW, outH := concatImagesVertically(imgs)

		scaleW := float32(imgViewWidth) / float32(outW)
		scaleH := float32(imgViewHeight) / float32(outH)
		scale := scaleW
		if scaleH < scaleW {
			scale = scaleH
		}
		if scale > 1.0 {
			scale = 1.0
		}
		newW := float32(outW) * scale
		newH := float32(outH) * scale

		imgResource.Image = outImg
		imgResource.SetMinSize(fyne.NewSize(newW, newH))
		imgResource.Resize(fyne.NewSize(newW, newH))
		imgResource.Refresh()
		imgBox.Resize(fyne.NewSize(float32(imgViewWidth), float32(imgViewHeight)))

		buf := new(bytes.Buffer)
		if err := png.Encode(buf, outImg); err == nil {
			_ = ioutil.WriteFile(TMP_FILE, buf.Bytes(), 0644)
		}
	}

	input.OnChanged = func(text string) {
		updateImage(text)
	}

	imgClipboardToTextBtn := widget.NewButton("Translate Loenese from clipboard to English", func() {
		err := saveClipboardImage(TMP_FILE)
		if err != nil {
			dialog.ShowError(err, myWindow)
			return
		}
		imgFile, err := os.Open(TMP_FILE)
		if err != nil {
			dialog.ShowError(err, myWindow)
			return
		}
		defer imgFile.Close()
		img, err := png.Decode(imgFile)
		if err != nil {
			dialog.ShowError(err, myWindow)
			return
		}
		lineImgs := splitImageByLine(img, IMG_HEIGHT)
		texts, err := ocrLinesParallel(lineImgs, port)
		if err != nil {
			dialog.ShowError(err, myWindow)
			return
		}
		finalText := strings.Join(texts, "\n")
		input.SetText(finalText)
		updateImage(finalText)
	})

	copyImgBtn := widget.NewButton("Copy Loenese", func() {
		f, err := os.Open(TMP_FILE)
		if err != nil {
			dialog.ShowError(err, myWindow)
			return
		}
		defer f.Close()
		if err := clipboard.Write(f); err != nil {
			dialog.ShowError(err, myWindow)
		}
	})

	saveImgBtn := widget.NewButton("Save Loenese", func() {
		saveDialog := dialog.NewFileSave(func(writer fyne.URIWriteCloser, err error) {
			if err != nil || writer == nil {
				return
			}
			defer writer.Close()
			imgFile, err := os.Open(TMP_FILE)
			if err != nil {
				dialog.ShowError(err, myWindow)
				return
			}
			defer imgFile.Close()
			_, err = io.Copy(writer, imgFile)
			if err != nil {
				dialog.ShowError(err, myWindow)
			}
		}, myWindow)
		saveDialog.SetFileName("text_image.png")
		saveDialog.Show()
	})

	clearBtn := widget.NewButton("Clear Input", func() {
		input.SetText("")
	})

	translateFromFileBtn := widget.NewButton("Translate from file to Leonese", func() {
		openDialog := dialog.NewFileOpen(func(reader fyne.URIReadCloser, err error) {
			if err != nil || reader == nil {
				return
			}
			defer reader.Close()
			img, _, err := image.Decode(reader)
			if err != nil {
				dialog.ShowError(err, myWindow)
				return
			}
			lineImgs := splitImageByLine(img, IMG_HEIGHT)
			texts, err := ocrLinesParallel(lineImgs, port)
			if err != nil {
				dialog.ShowError(err, myWindow)
				return
			}
			finalText := strings.Join(texts, "\n")
			input.SetText(finalText)
			updateImage(finalText)
		}, myWindow)
		openDialog.SetFilter(storage.NewExtensionFileFilter([]string{".png", ".jpg", ".jpeg", ".bmp"}))
		openDialog.Show()
	})

	myWindow.SetContent(container.NewVBox(
		widget.NewLabel("English-Loenese"),
		input,
		container.NewHBox(
			copyImgBtn,
			saveImgBtn,
			clearBtn,
		),
		container.NewHBox(
			imgClipboardToTextBtn,
		),
        container.NewHBox(
            translateFromFileBtn,
        ),
		imgBox,
	))

	myWindow.ShowAndRun()
}
