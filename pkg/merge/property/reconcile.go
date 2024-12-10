package property

import (
	"io/ioutil"
	"log"
	"path/filepath"
	"regexp"
	"strings"

	"federate/pkg/diff"
	"federate/pkg/federated"
	"federate/pkg/java"
	"federate/pkg/javast"
	"federate/pkg/manifest"
	"federate/pkg/merge/ledger"
	"github.com/sergi/go-diff/diffmatchpatch"
)

// 根据扫描的冲突情况进行调和，处理 .yml & .properties
func (cm *PropertyManager) Reconcile() (err error) {
	if err = cm.Prepare(); err != nil {
		return
	}

	// pass 1: 识别冲突
	conflicts := cm.identifyAllConflicts()
	if len(conflicts) == 0 {
		return
	}

	// pass 2: 注册表调和冲突，为冲突key增加前缀ns，对于 ConfigurationProperties/integralKey 整体处理
	conflictingKeysOfComponents := make(map[string][]string) // Group keys by component
	for key, components := range conflicts {
		for componentName, value := range components {
			conflictingKeysOfComponents[componentName] = append(conflictingKeysOfComponents[componentName], key)

			cm.r.SegregateProperty(componentName, Key(key), value)
		}
	}

	// pass 3: 修改对冲突key的引用
	for componentName, keys := range conflictingKeysOfComponents {
		keyMapping := make(map[string]string)
		for _, key := range keys {
			nsKey := Key(key).WithNamespace(componentName)
			keyMapping[key] = nsKey
		}

		component := *cm.m.ComponentByName(componentName)

		// 通过 Java AST 修改 Java 源代码里对冲突key的引用
		if err := javast.UpdatePropertyKeys(component, keyMapping); err != nil {
			return err
		}

		// 为 @RequestMapping 增加路径前缀
		componentServletContextPath := cm.servletContextPath[componentName]
		if componentServletContextPath != "" {
			contextPath := filepath.Clean("/" + strings.Trim(componentServletContextPath, "/"))
			if err := cm.segregateRequestMapping(component, contextPath); err != nil {
				return err
			}
		}

		// 修改 XML 里对冲突 key 的引用
		if err := cm.updateXMLPropertyReference(component, keys); err != nil {
			return err
		}
	}

	// pass 4: 合并到目标文件
	if cm.writeTarget {
		if err = cm.writeTargetFiles(); err != nil {
			return
		}
	}

	log.Printf("Source code rewritten, @RequestMapping: %d, @Value: %d, @ConfigurationProperties: %d",
		cm.result.RequestMapping, cm.result.KeyPrefixed, cm.result.ConfigurationProperties)
	return
}

// 修改 XML 里对冲突 key 的引用
func (cm *PropertyManager) updateXMLPropertyReference(c manifest.ComponentInfo, conflictKeys []string) error {
	keyRegexes := make([]*regexp.Regexp, len(conflictKeys))
	for i, key := range conflictKeys {
		keyRegexes[i] = P.createXMLPropertyReferenceRegex(key)
	}

	xmlChan, _ := java.ListXMLFilesAsync(c.RootDir())
	for f := range xmlChan {
		content, err := ioutil.ReadFile(f.Path)
		if err != nil {
			return err
		}

		oldContent := string(content)
		newContent := oldContent
		changed := false

		for i, regex := range keyRegexes {
			matches := regex.FindAllStringSubmatchIndex(oldContent, -1)
			if len(matches) > 0 {
				changed = true
				newContent = regex.ReplaceAllStringFunc(newContent, func(match string) string {
					newKey := Key(conflictKeys[i]).WithNamespace(c.Name)
					// do the update
					replaced := cm.transformXMLPropertyKeyReference(match, conflictKeys[i], newKey)
					dmp := diffmatchpatch.New()
					diffs := dmp.DiffMain(match, replaced, false)
					if cm.debug {
						log.Printf("[%s] Transforming %s\n%s", c.Name, f.Path, dmp.DiffPrettyText(diffs))
					}
					return replaced
				})
			}
		}

		if changed {
			// newContent 是[多次]叠加所有替换后的完整文件内容
			if err = ioutil.WriteFile(f.Path, []byte(newContent), f.Info.Mode()); err != nil {
				return err
			}
		}
	}

	return nil
}

func (cm *PropertyManager) transformXMLPropertyKeyReference(match, key, newKey string) string {
	parts := strings.Split(match, "${")
	for i := 1; i < len(parts); i++ {
		if strings.HasPrefix(parts[i], key) {
			parts[i] = newKey + strings.TrimPrefix(parts[i], key)
		}
	}
	return strings.Join(parts, "${")
}

// 修改 @RequestMapping path
func (cm *PropertyManager) segregateRequestMapping(c manifest.ComponentInfo, contextPath string) error {
	fileChan, _ := java.ListJavaMainSourceFilesAsync(c.RootDir())
	for f := range fileChan {
		content, err := ioutil.ReadFile(f.Path)
		if err != nil {
			return err
		}

		oldContent := string(content)
		newContent := P.requestMappingRegex.ReplaceAllStringFunc(oldContent, func(match string) string {
			submatches := P.requestMappingRegex.FindStringSubmatch(match)
			if len(submatches) == 4 {
				oldPath := filepath.Clean("/" + strings.TrimPrefix(submatches[2], "/"))
				if strings.HasPrefix(oldPath, contextPath) {
					return match // 如果已经包含，则不做改变
				}
				newPath := filepath.Join(contextPath, oldPath)
				return submatches[1] + newPath + submatches[3]
			}
			return match
		})

		if newContent != oldContent {
			ledger.Get().TransformRequestMapping(c.Name, "", contextPath)
			if cm.debug {
				diff.RenderUnifiedDiff(oldContent, newContent)
			}

			if err := ioutil.WriteFile(f.Path, []byte(newContent), f.Info.Mode()); err != nil {
				log.Fatalf("%v", err)
			}
			cm.result.RequestMapping++
		}
	}

	return nil
}

func (cm *PropertyManager) writeTargetFiles() (err error) {
	pn := filepath.Join(federated.GeneratedResourceBaseDir(cm.m.Main.Name), "application.properties")
	if err = cm.generateMergedPropertiesFile(pn); err != nil {
		return err
	}

	an := filepath.Join(federated.GeneratedResourceBaseDir(cm.m.Main.Name), "application.yml")
	if err = cm.generateMergedYamlFile(an); err != nil {
		return err
	}

	return
}
