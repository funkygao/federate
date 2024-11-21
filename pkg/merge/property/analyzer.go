package property

import (
	"fmt"
	"log"
	"path/filepath"

	"federate/pkg/manifest"
	"federate/pkg/util"
)

// 分析 .yml & .properties
func (cm *PropertyManager) Analyze() error {
	for _, component := range cm.m.Components {
		if err := cm.analyzeComponent(component); err != nil {
			return err
		}
	}

	// 解析所有引用
	cm.resolveAllReferences()

	// 应用保留key处理规则
	cm.applyReservedPropertyRules()
	return nil
}

func (cm *PropertyManager) analyzeComponent(component manifest.ComponentInfo) error {
	for _, baseDir := range component.Resources.BaseDirs {
		sourceDir := component.SrcDir(baseDir)

		// 分析 application.yml 和 application-{profile}.yml
		propertyFiles := []string{"application.yml"}
		if component.SpringProfile != "" {
			propertyFiles = append(propertyFiles, "application-"+component.SpringProfile+".yml")
		}
		// 加上用户指定的资源文件
		propertyFiles = append(propertyFiles, component.Resources.PropertySources...)

		for _, propertyFile := range propertyFiles {
			filePath := filepath.Join(sourceDir, propertyFile)
			if !util.FileExists(filePath) {
				log.Printf("[%s] Not found: %s", component.Name, filePath)
				continue
			}

			parser, supported := P.ParserByFileExt(filepath.Ext(propertyFile))
			if !supported {
				return fmt.Errorf("unsupported file type: %s", filePath)
			}

			if err := parser.Parse(filePath, component, cm); err != nil {
				return err
			}
		}
	}

	return nil
}
