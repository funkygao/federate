package property

import (
	"fmt"
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
	"federate/pkg/step"
	"github.com/sergi/go-diff/diffmatchpatch"
)

// 根据扫描的冲突情况进行调和，处理 .yml & .properties
func (pm *PropertyManager) Reconcile(bar step.Bar) (err error) {
	oldBarDesc := bar.State().Description
	defer bar.Describe(oldBarDesc)

	if err = pm.Prepare(); err != nil {
		return
	}
	bar.Add(1)

	// pass 1: 识别冲突
	bar.Describe(oldBarDesc + " Identifying conflicts")
	conflicts := pm.identifyAllConflicts()
	if len(conflicts) == 0 {
		return
	}
	bar.Add(2)

	// pass 2: 注册表调和冲突，为冲突key增加前缀ns，对于 ConfigurationProperties/integralKey 整体处理
	conflictingKeysOfComponents := make(map[string][]string) // Group keys by component
	for key, components := range conflicts {
		for componentName, value := range components {
			conflictingKeysOfComponents[componentName] = append(conflictingKeysOfComponents[componentName], key)

			pm.r.SegregateProperty(componentName, Key(key), value)
		}
	}

	// pass 3: 修改对冲突key的引用
	for componentName, keys := range conflictingKeysOfComponents {
		keyMapping := make(map[string]string)
		for _, key := range keys {
			nsKey := Key(key).WithNamespace(componentName)
			keyMapping[key] = nsKey
		}

		component := *pm.m.ComponentByName(componentName)

		// 通过 Java AST 修改 Java 源代码里对冲突key的引用
		javast.BacklogUpdatePropertyKeys(component, keyMapping)

		// 为 @RequestMapping 增加路径前缀
		componentServletContextPath := pm.servletContextPath[componentName]
		if componentServletContextPath != "" {
			bar.Describe(fmt.Sprintf("%s Transforming @RequestMapping for %s/", oldBarDesc, componentName))
			contextPath := filepath.Clean("/" + strings.Trim(componentServletContextPath, "/"))
			if err := pm.segregateRequestMapping(component, contextPath); err != nil {
				return err
			}
			bar.Add(50 / len(conflictingKeysOfComponents))
		}

		// 修改 XML 里对冲突 key 的引用
		bar.Describe(fmt.Sprintf("%s Transforming XML Files for %s/", oldBarDesc, componentName))
		if err := pm.updateXMLPropertyReference(component, keys); err != nil {
			return err
		}
		bar.Add(40 / len(conflictingKeysOfComponents))
	}

	// pass 4: 合并到目标文件
	if pm.writeTarget {
		if err = pm.writeTargetFiles(); err != nil {
			return
		}
		bar.Add(3)
	}

	bar.Describe(oldBarDesc)

	log.Printf("Source code rewritten, @RequestMapping: %d, @Value: %d, @ConfigurationProperties: %d",
		pm.result.RequestMapping, pm.result.KeyPrefixed, pm.result.ConfigurationProperties)
	return
}

// 修改 XML 里对冲突 key 的引用
func (pm *PropertyManager) updateXMLPropertyReference(c manifest.ComponentInfo, conflictKeys []string) error {
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

		// 对于每个冲突key，对该XML进行叠加式 prop key reference 替换
		for i, regex := range keyRegexes {
			matches := regex.FindAllStringSubmatchIndex(oldContent, -1)
			if len(matches) > 0 {
				changed = true

				newContent = regex.ReplaceAllStringFunc(newContent, func(match string) string {
					// do the replace
					newKey := Key(conflictKeys[i]).WithNamespace(c.Name)
					replaced := pm.transformXMLPropertyKeyReference(regex, match, conflictKeys[i], newKey)

					if pm.debug {
						dmp := diffmatchpatch.New()
						diffs := dmp.DiffMain(match, replaced, false)
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

func (pm *PropertyManager) transformXMLPropertyKeyReference(re *regexp.Regexp, match, key, newKey string) string {
	if re == nil {
		re = P.createXMLPropertyReferenceRegex(key)
	}
	return re.ReplaceAllString(match, "${1}"+newKey+"$2}")
}

// 修改 @RequestMapping path
func (pm *PropertyManager) segregateRequestMapping(c manifest.ComponentInfo, contextPath string) error {
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
			if pm.debug {
				diff.RenderUnifiedDiff(oldContent, newContent)
			}

			if err := ioutil.WriteFile(f.Path, []byte(newContent), f.Info.Mode()); err != nil {
				log.Fatalf("%v", err)
			}
			pm.result.RequestMapping++
		}
	}

	return nil
}

func (pm *PropertyManager) writeTargetFiles() (err error) {
	pn := filepath.Join(federated.GeneratedResourceBaseDir(pm.m.Main.Name), "application.properties")
	if err = pm.generateMergedPropertiesFile(pn); err != nil {
		return err
	}

	an := filepath.Join(federated.GeneratedResourceBaseDir(pm.m.Main.Name), "application.yml")
	if err = pm.generateMergedYamlFile(an); err != nil {
		return err
	}

	return
}
