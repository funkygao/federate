package compliance

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"federate/pkg/manifest"
)

func CheckComponentsCompliance(m *manifest.Manifest) {
	for _, component := range m.Components {
		for _, baseDir := range component.Resources.BaseDirs {
			sourceDir := filepath.Join(component.Name, baseDir)
			err := filepath.Walk(sourceDir, func(path string, info os.FileInfo, err error) error {
				if err != nil {
					return err
				}
				if !info.IsDir() {
					checkFileCompliance(path)
				}
				return nil
			})
			if err != nil {
				fmt.Printf("Error checking component %s: %v\n", component.Name, err)
			}
		}
	}
}

func checkFileCompliance(path string) {
	// 示例检查：文件名不能包含空格
	if strings.Contains(filepath.Base(path), " ") {
		fmt.Printf("Compliance issue: File %s contains spaces in its name\n", path)
	}

	// 你可以在这里添加更多的检查逻辑
}
