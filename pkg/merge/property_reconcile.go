package merge

import (
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"

	"federate/pkg/concurrent"
	"federate/pkg/diff"
	"federate/pkg/java"
	"federate/pkg/manifest"
	"github.com/sergi/go-diff/diffmatchpatch"
)

type PropertySourcesReconcileReport struct {
	KeyPrefixed             int
	RequestMapping          int
	ConfigurationProperties int
}

// 根据扫描的冲突情况进行调和，处理 .yml & .properties
func (cm *PropertyManager) Reconcile(dryRun bool) (result PropertySourcesReconcileReport, err error) {
	// pass 1: 识别冲突
	conflicts := cm.IdentifyAllConflicts()
	if len(conflicts) == 0 {
		return
	}

	// pass 2:
	conflictingKeysOfComponents := make(map[string][]string) // Group keys by component
	for key, components := range conflicts {
		for componentName, value := range components {
			conflictingKeysOfComponents[componentName] = append(conflictingKeysOfComponents[componentName], key)

			cm.updateResolvedProperties(componentName, Key(key), value)
		}
	}

	// pass 3: 创建并发任务对 Component 源代码进行插桩改写
	executor := concurrent.NewParallelExecutor(runtime.NumCPU())
	executor.SetName("Overwrite Java/XML conflicted property references & @RequestMapping & @ConfigurationProperties")
	for componentName, keys := range conflictingKeysOfComponents {
		executor.AddTask(&reconcileTask{
			cm:                 cm,
			component:          cm.m.ComponentByName(componentName),
			keys:               keys,
			dryRun:             dryRun,
			servletContextPath: cm.servletContextPath[componentName],
			result:             reconcileTaskResult{},
		})
	}

	errors := executor.Execute()
	if len(errors) > 0 {
		err = errors[0] // 返回第一个遇到的错误
	}

	for _, task := range executor.Tasks() {
		reconcileTask := task.(*reconcileTask)
		result.KeyPrefixed += reconcileTask.result.keyPrefixed
		result.RequestMapping += reconcileTask.result.requestMapping
		result.ConfigurationProperties += reconcileTask.result.configurationProperties
	}

	return
}

func (pm *PropertyManager) updateResolvedProperties(componentName string, key Key, value interface{}) {
	strKey := string(key)
	originalSource := pm.resolvedProperties[componentName][strKey]

	// 已经占位符替换的，要恢复占位符
	newOriginalString := originalSource.OriginalString
	if strings.Contains(newOriginalString, "${") {
		// 更新 OriginalString 中的引用 ${foo} => ${component1.foo}
		newOriginalString = pm.updateReferencesInString(originalSource.OriginalString, componentName)
		if !pm.silent {
			log.Printf("[%s] Key=%s Ref Updated: %s => %s", componentName, strKey, originalSource.OriginalString, newOriginalString)
		}
	}

	// 修改 resolvedProperties，写盘使用
	if integralKey := pm.getConfigurationPropertiesPrefix(strKey); integralKey != "" {
		pm.handleConfigurationProperties(componentName, integralKey)
	} else {
		pm.handleRegularProperty(componentName, key, value, newOriginalString, originalSource)
	}
}

func (pm *PropertyManager) handleConfigurationProperties(componentName, configPropPrefix string) {
	// key 是 ConfigurationProperties 的一部分，为所有相关的 key 添加命名空间
	for subKey := range pm.resolvedProperties[componentName] {
		if strings.HasPrefix(subKey, configPropPrefix) {
			nsKey := Key(subKey).WithNamespace(componentName)
			pm.resolvedProperties[componentName][nsKey] = PropertySource{
				Value:          pm.resolvedProperties[componentName][subKey].Value,
				OriginalString: pm.resolvedProperties[componentName][subKey].OriginalString,
				FilePath:       pm.resolvedProperties[componentName][subKey].FilePath,
			}
		}
	}
}

func (pm *PropertyManager) handleRegularProperty(componentName string, key Key, value interface{}, newOriginalString string, originalSource PropertySource) {
	nsKey := key.WithNamespace(componentName)
	pm.resolvedProperties[componentName][nsKey] = PropertySource{
		Value:          value,
		OriginalString: newOriginalString,
		FilePath:       originalSource.FilePath,
	}
}

func (cm *PropertyManager) updateReferencesInString(s, componentName string) string {
	return os.Expand(s, func(key string) string {
		return "${" + componentName + "." + key + "}"
	})
}

type reconcileTask struct {
	cm *PropertyManager

	component          *manifest.ComponentInfo
	keys               []string
	dryRun             bool
	servletContextPath string
	result             reconcileTaskResult
}

type reconcileTaskResult struct {
	keyPrefixed             int
	requestMapping          int
	configurationProperties int
}

func (t *reconcileTask) Execute() error {
	// 为Java源代码里这些key的引用增加组件名称前缀作为ns
	if err := t.namespaceKeyReferences(java.IsJavaMainSource, P.createJavaRegex); err != nil {
		return err
	}

	// 为xml里这些key的引用增加组件名称前缀作为ns
	if err := t.namespaceKeyReferences(java.IsXml, P.createXmlRegex); err != nil {
		return err
	}

	// 处理 @ConfigurationProperties
	if err := t.namespaceKeyReferences(java.IsJavaMainSource, P.createConfigurationPropertiesRegex); err != nil {
		return err
	}

	// 解决 server.servlet.context-path 冲突：修改Java源代码
	if !t.dryRun && t.servletContextPath != "" {
		if err := t.updateRequestMappings(); err != nil {
			return err
		}
	}

	return nil
}

func (t *reconcileTask) namespaceKeyReferences(fileFilter func(os.FileInfo, string) bool, createRegex func(string) *regexp.Regexp) error {
	keyRegexes := make([]*regexp.Regexp, len(t.keys))
	for i, key := range t.keys {
		keyRegexes[i] = createRegex(key)
	}

	return filepath.Walk(t.component.RootDir(), func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if fileFilter(info, path) {
			content, err := ioutil.ReadFile(path)
			if err != nil {
				return err
			}

			oldContent := string(content)
			newContent := oldContent
			changed := false

			for i, regex := range keyRegexes {
				matches := regex.FindAllStringSubmatchIndex(newContent, -1)
				if len(matches) > 0 {
					changed = true
					newContent = regex.ReplaceAllStringFunc(newContent, func(match string) string {
						newKey := Key(t.keys[i]).WithNamespace(t.component.Name)
						replaced := t.replaceKeyInMatch(match, t.keys[i], newKey)
						dmp := diffmatchpatch.New()
						diffs := dmp.DiffMain(match, replaced, false)
						log.Printf("%s", dmp.DiffPrettyText(diffs))
						return replaced
					})
					t.result.keyPrefixed++
					if strings.Contains(regex.String(), "@ConfigurationProperties") {
						t.result.configurationProperties++
					} else {
						t.result.keyPrefixed++
					}
				}
			}

			if changed && !t.dryRun {
				err = ioutil.WriteFile(path, []byte(newContent), info.Mode())
				if err != nil {
					return err
				}
				log.Printf("↖ %s", path)
			}
		}
		return nil
	})
}

func (t *reconcileTask) replaceKeyInMatch(match, key, newKey string) string {
	if strings.Contains(match, "@ConfigurationProperties") {
		return strings.Replace(match, `"`+key+`"`, `"`+newKey+`"`, 1)
	}
	return strings.Replace(match, "${"+key, "${"+newKey, 1)
}

func (t *reconcileTask) updateRequestMappings() error {
	contextPath := filepath.Clean("/" + strings.Trim(t.servletContextPath, "/"))
	return filepath.Walk(t.component.RootDir(), func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if java.IsJavaMainSource(info, path) {
			content, err := ioutil.ReadFile(path)
			if err != nil {
				return err
			}

			oldContent := string(content)
			newContent := t.updateRequestMappingInFile(oldContent, contextPath)
			if newContent != oldContent {
				if !t.dryRun {
					diff.RenderUnifiedDiff(oldContent, newContent)

					err = ioutil.WriteFile(path, []byte(newContent), info.Mode())
					if err != nil {
						return err
					}
					log.Printf("↖ %s", path)
					t.result.requestMapping++
				}
			}
		}

		return nil
	})
}

func (t *reconcileTask) updateRequestMappingInFile(content, contextPath string) string {
	return P.requestMappingRegex.ReplaceAllStringFunc(content, func(match string) string {
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
}
