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
func (cm *PropertyManager) ReconcileConflicts(dryRun bool) (result PropertySourcesReconcileReport, err error) {
	conflicts := cm.IdentifyAllConflicts()
	if len(conflicts) == 0 {
		return
	}

	// Group keys by component
	conflictingKeysOfComponents := make(map[string][]string)
	for key, components := range conflicts {
		for componentName, value := range components {
			conflictingKeysOfComponents[componentName] = append(conflictingKeysOfComponents[componentName], key)

			cm.resolveConflict(componentName, Key(key), value)
		}
	}

	executor := concurrent.NewParallelExecutor(runtime.NumCPU())
	executor.SetName("Overwrite Java/XML conflicted property references & RequestMapping")
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
