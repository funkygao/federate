package merge

import (
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"runtime"

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
	conflictKeys := cm.IdentifyYamlFileConflicts()
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
			cm.mergedYaml[prefixedKey] = value
			//delete(cm.mergedYaml, key) 原有的key不能删除：第三方包内部，可能在使用该 key

			cellData = append(cellData, []string{prefixedKey, util.Truncate(fmt.Sprintf("%v", value), 60)})
		}
	}

	header := []string{"New Key", "Value"}
	tablerender.DisplayTable(header, cellData, false, -1)
	log.Printf("Reconciled %d conflicting keys into %d keys", len(conflictKeys), len(cellData))

	executor := concurrent.NewParallelExecutor(runtime.NumCPU())
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
	// TODO yaml 与 properties 的相互引用：key增加了ns，则需要为相应引用处也修改
	// important.properties:  master.ds2.mysql.url = ${datasource.master.ds2.mysql.url} # 安全要求，代码库里important.properties不能明文保存账钥等
	// application.yml:       datasource.master.ds2.mysql.url: jdbc:mysql://1.1.1.1:3306/foo
	// 现在yml里该key冲突，datasource.master.ds2.mysql.url -> stock.datasource.master.ds2.mysql.url，需要 resolve master.ds2.mysql.url
	// 这就带来新的问题：如果一次扫描发现冲突，就会造成不同component master.ds2.mysql.url之前引用的key是相同的，而reconcile后就不同了
	// 因此，扫描时，要发现所有的 value reference 展开，把 yaml 与 properties 统一处理后，才能发现冲突
	// 我建议：在 reconcile 前，注册冲突前，捕获属性引用，之后才注册冲突，这样更方便代码实现，而且不漏
	// 即：把所有的 reference 都解析完了，再发现冲突
	// TODO end

	// 为Java源代码里这些key的引用增加组件名称前缀作为ns
	if err := t.prefixKeyReferences(java.IsJavaMainSource, t.cm.createJavaRegex); err != nil {
		return err
	}

	// 为xml里这些key的引用增加组件名称前缀作为ns
	if err := t.prefixKeyReferences(java.IsXml, t.cm.createXmlRegex); err != nil {
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
			newContent := t.cm.updateRequestMappingInFile(oldContent, t.servletContextPath)
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
						replaced := t.cm.replaceKeyInMatch(match, t.keys[i], t.prefix)
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
