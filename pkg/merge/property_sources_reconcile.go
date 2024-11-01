package merge

import (
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"regexp"

	"federate/pkg/diff"
	"federate/pkg/java"
	"federate/pkg/manifest"
	"github.com/sergi/go-diff/diffmatchpatch"
)

type reconcileTask struct {
	cm *PropertySourcesManager

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
	// 为Java源代码里这些key的引用增加组件名称前缀
	if err := t.prefixKeyReferences(t.component.RootDir(), t.keys, t.prefix, t.dryRun, java.IsJavaMainSource, t.cm.createJavaRegex); err != nil {
		return err
	}

	// 为xml里这些key的引用增加组件名称前缀
	if err := t.prefixKeyReferences(t.component.TargetResourceDir(), t.keys, t.prefix, t.dryRun, java.IsXml, t.cm.createXmlRegex); err != nil {
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

func (t *reconcileTask) prefixKeyReferences(baseDir string, keys []string, prefix string, dryRun bool, fileFilter func(os.FileInfo, string) bool, createRegex func(string) *regexp.Regexp) error {
	keyRegexes := make([]*regexp.Regexp, len(keys))
	for i, key := range keys {
		keyRegexes[i] = createRegex(key)
	}

	return filepath.Walk(baseDir, func(path string, info os.FileInfo, err error) error {
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
						replaced := t.cm.replaceKeyInMatch(match, keys[i], prefix)
						dmp := diffmatchpatch.New()
						diffs := dmp.DiffMain(match, replaced, false)
						log.Printf("%s", dmp.DiffPrettyText(diffs))
						return replaced
					})
					t.result.keyPrefixed++
				}
			}

			if changed && !dryRun {
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
