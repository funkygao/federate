package merge

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"federate/pkg/java"
	"federate/pkg/manifest"
	"federate/pkg/util"
)

var (
	resourceExtensions = map[string]struct{}{
		".json":       {}, // private ConfigMaps
		".xml":        {}, // beans, mybatis-config
		".properties": {}, // i18N message bundles, properties
		".yml":        {}, // spring boot
		".html":       {}, // email templates
	}

	// 使用不区分大小写的正则表达式来匹配 Spring 配置文件
	yamlFilePattern = regexp.MustCompile(`(?i)^application(-[a-zA-Z0-9\-]+)?\.(yml)$`)
)

type ResourceManager struct {
	ExtensionCount    map[string]int
	metaInfCount      int
	FilePaths         map[string]string
	totalFileCount    map[string]int
	resourceFileNames map[string]map[string]struct{}
}

func NewResourceManager() *ResourceManager {
	ExtensionCount := make(map[string]int)
	for ext := range resourceExtensions {
		ExtensionCount[ext] = 0
	}
	return &ResourceManager{
		ExtensionCount:    ExtensionCount,
		FilePaths:         make(map[string]string),
		totalFileCount:    make(map[string]int),
		resourceFileNames: make(map[string]map[string]struct{}),
	}
}

// RecursiveFederatedCopyResources copy component resource files to target system src/main/resources/federated/{component}
func (rm *ResourceManager) RecursiveFederatedCopyResources(m *manifest.Manifest) error {
	for _, component := range m.Components {
		for _, baseDir := range component.Resources.BaseDirs {
			sourceDir := component.SrcDir(baseDir)
			targetDir := component.TargetResourceDir()

			if err := os.MkdirAll(targetDir, 0755); err != nil {
				log.Fatalf("Error creating directory for component %s: %v", component.Name, err)
			}

			log.Printf("[%s:%s] Federated Copying %s -> %s", component.Name, component.SpringProfile, sourceDir, targetDir)
			if err := rm.federatedCopyResources(sourceDir, targetDir, component, m); err != nil {
				log.Fatalf("Error copying resources for component %s: %v", component.Name, err)
			}
		}
	}
	return nil
}

func (rm *ResourceManager) federatedCopyResources(sourceDir, targetDir string, component manifest.ComponentInfo, m *manifest.Manifest) error {
	err := filepath.Walk(sourceDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if ignored := m.IgnoreResourceSrcFile(info, component); ignored {
			log.Printf("[%s:%s] Skipped federatedIgnore: %s", component.Name, component.SpringProfile, info.Name())
			return nil
		}

		relPath, err := filepath.Rel(sourceDir, path)
		if err != nil {
			return err
		}
		targetPath := filepath.Join(targetDir, relPath)
		if info.IsDir() {
			return os.MkdirAll(targetPath, info.Mode())
		}
		if (java.IsResourceFile(info, path) || java.IsMetaInfFile(info, path)) && !java.IsSpringYamlFile(info, path) {
			if java.IsMetaInfFile(info, path) {
				rm.metaInfCount++
			} else {
				ext := strings.ToLower(filepath.Ext(path))
				rm.ExtensionCount[ext]++
				rm.totalFileCount[ext]++
			}
			rm.FilePaths[relPath] = component.Name // 记录文件的相对路径和组件名称
			if _, exists := rm.resourceFileNames[relPath]; !exists {
				rm.resourceFileNames[relPath] = make(map[string]struct{})
			}
			rm.resourceFileNames[relPath][component.Name] = struct{}{}
			//log.Printf("[%s:%s] Copying %s", component.Name, component.SpringProfile, info.Name())
			return util.CopyFile(path, targetPath)
		}
		return nil
	})

	rm.ExtensionCount[metaInf] = rm.metaInfCount

	return err
}

// RecursiveFlatCopyResources copy specified resource files to target system src/main/resources
func (rm *ResourceManager) RecursiveFlatCopyResources(m *manifest.Manifest) error {
	copiedFiles := make(map[string]struct{}) // key is target filename
	targetDir := m.TargetResourceDir()
	for _, pattern := range m.Main.Reconcile.Resources.FlatCopy {
		for _, component := range m.Components {
			for _, baseDir := range component.Resources.BaseDirs {
				sourceDir := component.SrcDir(baseDir)
				err := filepath.Walk(sourceDir, func(path string, info os.FileInfo, err error) error {
					if err != nil {
						return err
					}

					relPath, err := filepath.Rel(sourceDir, path)
					if err != nil {
						return err
					}

					// Check if the file matches the pattern
					match, err := filepath.Match(pattern, filepath.Base(path))
					if err != nil {
						return err
					}

					if match && !info.IsDir() {
						targetPath := filepath.Join(targetDir, relPath)
						if err := os.MkdirAll(filepath.Dir(targetPath), 0755); err != nil {
							return err
						}

						if err := util.CopyFile(path, targetPath); err != nil {
							return err
						}

						if _, present := copiedFiles[relPath]; present {
							return fmt.Errorf("Flat copy %s conflicts", relPath)
						}

						copiedFiles[relPath] = struct{}{}

						log.Printf("[%s:%s] Flat Copied %s", component.Name, component.SpringProfile, relPath)
					}

					return nil
				})
				if err != nil {
					return err
				}
			}
		}
	}

	return nil
}

func (rm *ResourceManager) GetTotalFileCount() map[string]int {
	return rm.totalFileCount
}

func (rm *ResourceManager) GetResourceFileNames() map[string]map[string]struct{} {
	return rm.resourceFileNames
}
