package cmd

import (
	"log"
	"os"
	"path/filepath"
	"text/template"

	"federate/pkg/manifest"
	"github.com/spf13/cobra"
)

func generateFileFromTmpl(templatePath, outputPath string, data interface{}) {
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
	tmpl, err := template.New(tmplName).Funcs(funcMap).ParseFS(templates, templatePath)
	if err != nil {
		log.Fatalf("Error parsing template: %v", err)
	}

	// 检查目录是否存在，如果不存在则创建
	outputDir := filepath.Dir(outputPath)
	if _, err := os.Stat(outputDir); os.IsNotExist(err) {
		err = os.MkdirAll(outputDir, 0755)
		if err != nil {
			log.Fatalf("Error creating directory: %v", err)
		}
	}

	file, err := os.Create(outputPath)
	if err != nil {
		log.Fatalf("Error creating file: %v", err)
	}
	defer file.Close()

	err = tmpl.Execute(file, data)
	if err != nil {
		log.Fatalf("Error executing template: %v", err)
	}
}

// addRequiredInputFlag adds the --input flag to the given command and marks it as required
func addRequiredInputFlag(cmd *cobra.Command) {
	cmd.Flags().StringVarP(&manifestFile, "input", "i", "", "Path to the manifest file")
	cmd.MarkFlagRequired("input")
}
