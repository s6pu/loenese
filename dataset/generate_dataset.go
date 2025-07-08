package main

import (
    "bufio"
    "fmt"
    "image"
    "image/color"
    "image/draw"
    "image/png"
    "io/ioutil"
    "log"
    "math/rand"
    "os"
    "path/filepath"
    "regexp"
    "strings"
    "time"

    "golang.org/x/image/font"
    "golang.org/x/image/font/opentype"
    "golang.org/x/image/math/fixed"
)

var allowedRegexp = regexp.MustCompile(`^[A-Za-z0-9!()'\";:/?., ]+$`)

// Checks if a string contains only allowed characters
func isAllowed(s string) bool {
    return allowedRegexp.MatchString(s)
}

// Loads and filters lines from a corpus file
func loadFilteredCorpus(path string, maxLines int) ([]string, error) {
    file, err := os.Open(path)
    if err != nil {
        return nil, err
    }
    defer file.Close()

    var filtered []string
    scanner := bufio.NewScanner(file)
    for scanner.Scan() {
        line := strings.TrimSpace(scanner.Text())
        if isAllowed(line) && len(line) > 0 && len(line) <= 32 {
            filtered = append(filtered, line)
            if len(filtered) >= maxLines {
                break
            }
        }
    }
    return filtered, scanner.Err()
}

// Renders a single text line to an image with fixed height
func renderTextToImage(text, fontPath string, fontSize float64, imgHeight int) (image.Image, int, error) {
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
    textWidth := dr.MeasureString(text).Ceil() + 20 // 10px padding left/right
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

func itoa(i int) string {
    return fmt.Sprintf("%d", i)
}

// Generates a random string from the allowed charset
func randomString(length int, charset string) string {
    b := make([]byte, length)
    for i := range b {
        b[i] = charset[rand.Intn(len(charset))]
    }
    return string(b)
}

func main() {
    fontPath := "../assets/Loen.otf"
    fontSize := 48.0
    imgHeight := 64
    outDir := "images"
    os.MkdirAll(outDir, 0755)

    corpusSamples, err := loadFilteredCorpus("../data/corpus.txt", 100000)
    if err != nil {
        log.Fatal("Failed to load corpus:", err)
    }

    // Collect unique samples
    sampleSet := make(map[string]struct{})
    for _, s := range corpusSamples {
        sampleSet[s] = struct{}{}
    }

    // Add all letters, digits, punctuation, and some base words
    upper := "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    lower := "abcdefghijklmnopqrstuvwxyz"
    digits := "0123456789"
    punct := "!()'\";:/?.,"
    baseWords := []string{"Hello", "World", "Test", "Mix", "OK", "Yes", "No"}

    for _, s := range baseWords {
        sampleSet[s] = struct{}{}
    }
    sampleSet[upper] = struct{}{}
    sampleSet[lower] = struct{}{}
    sampleSet[digits] = struct{}{}
    sampleSet[punct] = struct{}{}

    // Generate random strings to reach the desired dataset size
    rand.Seed(time.Now().UnixNano())
    allowedChars := upper + lower + digits + punct + " "
    for len(sampleSet) < 100000 {
        l := rand.Intn(29) + 4 // string length: 4 to 32
        s := randomString(l, allowedChars)
        sampleSet[s] = struct{}{}
    }

    // Convert set to slice
    samples := make([]string, 0, 100000)
    for s := range sampleSet {
        samples = append(samples, s)
    }

    // Write images and ground truth labels
    gtFile, err := os.Create(filepath.Join(outDir, "labels.txt"))
    if err != nil {
        log.Fatalf("Failed to create labels.txt: %v", err)
    }
    defer gtFile.Close()

    for i, text := range samples {
        img, _, err := renderTextToImage(text, fontPath, fontSize, imgHeight)
        if err != nil {
            log.Printf("Failed to generate image for line %d: %v", i+1, err)
            continue
        }
        imgFile := filepath.Join(outDir, "img_"+itoa(i+1)+".png")

        f, err := os.Create(imgFile)
        if err != nil {
            log.Printf("Failed to save image: %v", err)
            continue
        }
        if err := png.Encode(f, img); err != nil {
            log.Printf("PNG encode error: %v", err)
        }
        f.Close()

        gtFile.WriteString(fmt.Sprintf("img_%d.png %s\n", i+1, text))
    }
}
