package fs

import (
	"bytes"
	"embed"
	"encoding/base64"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"text/template"

	"federate/pkg/diff"
	"federate/pkg/manifest"
	"federate/pkg/util"
)

//go:embed templates/*
var FS embed.FS

func GenerateExecutableFileFromTmpl(templatePath, outputPath string, data any) (overwrite bool) {
	overwrite = GenerateFileFromTmpl(templatePath, outputPath, data)
	err := os.Chmod(outputPath, 0755)
	if err != nil {
		log.Fatalf("Error setting file permissions: %v", err)
	}
	return
}

func GenerateFileFromTmpl(templatePath, outputPath string, data any) (overwrite bool) {
	// 创建一个 FuncMap 来注册自定义函数
	funcMap := template.FuncMap{
		"HasFeature": func(feature string) bool {
			m, ok := data.(*manifest.Manifest)
			if !ok {
				return false
			}
			return m.HasFeature(feature)
		},
	}

	// 解析嵌入的模板文件并应用 FuncMap
	tmpl, err := template.New(filepath.Base(templatePath)).Funcs(funcMap).ParseFS(FS, templatePath)
	if err != nil {
		log.Fatalf("Error parsing template: %v", err)
	}

	var (
		output     io.Writer
		oldContent []byte
	)
	if outputPath == "" {
		// 如果 outputPath 为空，使用 os.Stdout
		output = os.Stdout
	} else {
		// 检查目录是否存在，如果不存在则创建
		outputDir := filepath.Dir(outputPath)
		if _, err := os.Stat(outputDir); os.IsNotExist(err) {
			err = os.MkdirAll(outputDir, 0755)
			if err != nil {
				log.Fatalf("Error creating directory: %v", err)
			}
		}

		// 检测目标文件是否存在
		if util.FileExists(outputPath) {
			overwrite = true
			oldContent, _ = ioutil.ReadFile(outputPath)
		}
		file, err := os.Create(outputPath)
		if err != nil {
			log.Fatalf("Error creating file: %v", err)
		}
		defer file.Close()
		output = file
	}

	var buf bytes.Buffer
	multiOutput := io.MultiWriter(&buf, output)

	err = tmpl.Execute(multiOutput, data)
	if err != nil {
		log.Fatalf("Error executing template: %v", err)
	}

	if overwrite {
		oldContentStr := string(oldContent)
		newContentStr := buf.String()
		if oldContentStr != newContentStr {
			log.Printf("Overwrite %s", outputPath)

			diff.ShowDiffLineByLine(oldContentStr, newContentStr)
		}
	} else if outputPath != "" {
		log.Printf("Generated %s", outputPath)
	}
	return
}

func ParseTemplate(templatePath string, data any) bytes.Buffer {
	// 解析嵌入的模板文件
	tmplName := filepath.Base(templatePath)
	tmpl, err := template.New(tmplName).ParseFS(FS, templatePath)
	if err != nil {
		log.Fatalf("Error parsing template: %v", err)
	}

	// 使用 bytes.Buffer 来存储渲染结果
	var buf bytes.Buffer

	// 执行模板
	err = tmpl.Execute(&buf, data)
	if err != nil {
		log.Fatalf("Error parsing template: %v", err)
	}

	return buf
}

func ParseTemplateToString(templatePath string, data any) string {
	buf := ParseTemplate(templatePath, data)
	return buf.String()
}

// IsRunningInITerm2 检查当前是否在 iTerm2 中运行
func IsRunningInITerm2() bool {
	return os.Getenv("TERM_PROGRAM") == "iTerm.app"
}

func DisplayJPGInITerm2(templatePath string) {
	if !IsRunningInITerm2() {
		return
	}

	buf := ParseTemplate(templatePath, nil)

	// 将图像数据编码为 base64
	encodedImage := base64.StdEncoding.EncodeToString(buf.Bytes())

	// 构造 iTerm2 图像显示协议的转义序列
	fmt.Printf("\033]1337;File=inline=1;width=auto;height=auto:%s\a\n", encodedImage)
}
