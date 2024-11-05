package merge

import (
	"fmt"
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
	"federate/pkg/tablerender"
	"federate/pkg/util"
	"github.com/sergi/go-diff/diffmatchpatch"
)

type PropertySourcesReconcileReport struct {
	KeyPrefixed    int
	RequestMapping int
}

// 根据扫描的冲突情况进行调和，处理 .yml & .properties
func (cm *PropertyManager) ReconcileConflicts(dryRun bool) (result PropertySourcesReconcileReport, err error) {
	conflictKeys := cm.IdentifyAllConflicts()
	if len(conflictKeys) == 0 {
		return
	}

	// Group keys by component
	componentKeys := make(map[string][]string)
	var cellData [][]string
	for key, components := range conflictKeys {
		for componentName, value := range components {
			componentKeys[componentName] = append(componentKeys[componentName], key)

			prefixedKey := Key(key).WithNamespace(componentName)
			if value == nil {
				value = ""
			}

			// Update the resolvedProperties with the prefixed key, for .properties && .yml
			cm.resolvedProperties[componentName][prefixedKey] = PropertySource{
				Value:    value,
				FilePath: cm.resolvedProperties[componentName][key].FilePath,
			}

			//delete(cm.mergedYaml, key) 原有的key不能删除：第三方包内部，可能在使用该 key

			cellData = append(cellData, []string{prefixedKey, util.Truncate(fmt.Sprintf("%v", value), 60)})
		}
	}

	header := []string{"New Key", "Value"}
	tablerender.DisplayTable(header, cellData, false, -1)
	log.Printf("Reconciled %d conflicting keys into %d keys", len(conflictKeys), len(cellData))

	executor := concurrent.NewParallelExecutor(runtime.NumCPU())
	executor.SetName("Overwrite Java/XML conflicted property references & RequestMapping")
	for componentName, keys := range componentKeys {
		executor.AddTask(&reconcileTask{
			cm:                 cm,
			component:          cm.m.ComponentByName(componentName),
			keys:               keys,
			prefix:             Key("").NamespacePrefix(componentName),
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
	}

	return
}

type reconcileTask struct {
	cm *PropertyManager

	component          *manifest.ComponentInfo
	keys               []string
	prefix             string
	dryRun             bool
	servletContextPath string
	result             reconcileTaskResult
}

type reconcileTaskResult struct {
	keyPrefixed    int
	requestMapping int
}

func (t *reconcileTask) Execute() error {
	// 为Java源代码里这些key的引用增加组件名称前缀作为ns
	if err := t.prefixKeyReferences(java.IsJavaMainSource, t.createJavaRegex); err != nil {
		return err
	}

	// 为xml里这些key的引用增加组件名称前缀作为ns
	if err := t.prefixKeyReferences(java.IsXml, t.createXmlRegex); err != nil {
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

func (t *reconcileTask) createJavaRegex(key string) *regexp.Regexp {
	return regexp.MustCompile(`@Value\s*\(\s*"\$\{` + regexp.QuoteMeta(key) + `(:[^}]*)?\}"\s*\)`)
}

func (t *reconcileTask) createXmlRegex(key string) *regexp.Regexp {
	return regexp.MustCompile(`(value|key)="\$\{` + regexp.QuoteMeta(key) + `(:[^}]*)?\}"`)
}

func (t *reconcileTask) replaceKeyInMatch(match, key, prefix string) string {
	return strings.Replace(match, "${"+key, "${"+prefix+key, 1)
}

func (t *reconcileTask) updateRequestMappings() error {
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
			newContent := t.updateRequestMappingInFile(oldContent, t.servletContextPath)
			if newContent != oldContent {
				if !t.dryRun {
					diff.RenderUnifiedDiff(oldContent, newContent)

					err = ioutil.WriteFile(path, []byte(newContent), info.Mode())
					if err != nil {
						return err
					}
					log.Printf("%s", path)
					t.result.requestMapping++
				}
			}
		}

		return nil
	})
}

func (t *reconcileTask) updateRequestMappingInFile(content, contextPath string) string {
	return t.cm.requestMappingRegex.ReplaceAllStringFunc(content, func(match string) string {
		submatches := t.cm.requestMappingRegex.FindStringSubmatch(match)
		if len(submatches) == 4 {
			oldPath := submatches[2]
			newPath := filepath.Join(contextPath, oldPath)
			return submatches[1] + newPath + submatches[3]
		}
		return match
	})
}

func (t *reconcileTask) prefixKeyReferences(fileFilter func(os.FileInfo, string) bool, createRegex func(string) *regexp.Regexp) error {
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
						replaced := t.replaceKeyInMatch(match, t.keys[i], t.prefix)
						dmp := diffmatchpatch.New()
						diffs := dmp.DiffMain(match, replaced, false)
						log.Printf("%s", dmp.DiffPrettyText(diffs))
						return replaced
					})
					t.result.keyPrefixed++
				}
			}

			if changed && !t.dryRun {
				err = ioutil.WriteFile(path, []byte(newContent), info.Mode())
				if err != nil {
					return err
				}
				log.Printf("%s", path)
			}
		}
		return nil
	})
}
