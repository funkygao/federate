package fs

import (
	"embed"
	"io"
	"log"
	"os"
	"path/filepath"
	"text/template"

	"federate/pkg/manifest"
	"federate/pkg/util"
)

//go:embed templates/*
var FS embed.FS

func GenerateExecutableFileFromTmpl(templatePath, outputPath string, data interface{}) (overwrite bool) {
	overwrite = GenerateFileFromTmpl(templatePath, outputPath, data)
	err := os.Chmod(outputPath, 0755)
	if err != nil {
		log.Fatalf("Error setting file permissions: %v", err)
	}
	return
}

func GenerateFileFromTmpl(templatePath, outputPath string, data interface{}) (overwrite bool) {
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
	tmplName := filepath.Base(templatePath)
	tmpl, err := template.New(tmplName).Funcs(funcMap).ParseFS(FS, templatePath)
	if err != nil {
		log.Fatalf("Error parsing template: %v", err)
	}

	var output io.Writer
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
		}
		file, err := os.Create(outputPath)
		if err != nil {
			log.Fatalf("Error creating file: %v", err)
		}
		defer file.Close()
		output = file
	}

	err = tmpl.Execute(output, data)
	if err != nil {
		log.Fatalf("Error executing template: %v", err)
	}
	return
}
