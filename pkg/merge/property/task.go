package property

import (
	"context"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"federate/pkg/code"
	"federate/pkg/diff"
	"federate/pkg/java"
	"federate/pkg/manifest"
	"federate/pkg/merge/ledger"
	"github.com/sergi/go-diff/diffmatchpatch"
)

type reconcileTask struct {
	pm                 *PropertyManager
	c                  *manifest.ComponentInfo
	keys               []string
	servletContextPath string

	result ReconcileReport
}

func (t *reconcileTask) Execute() error {
	// 为Java源代码里这些key的引用增加组件名称前缀作为ns
	if err := t.namespaceKeyReferences(java.IsJavaMainSource, createJavaRegex); err != nil {
		return err
	}

	// 处理 @ConfigurationProperties
	if err := t.namespaceKeyReferences(java.IsJavaMainSource, createConfigurationPropertiesRegex); err != nil {
		return err
	}

	// 解决 server.servlet.context-path 冲突：修改Java源代码
	if t.servletContextPath != "" {
		if err := t.updateRequestMappings(); err != nil {
			return err
		}
	}

	// 为xml里这些key的引用增加组件名称前缀作为ns
	if err := t.namespaceKeyReferences(java.IsXML, createXmlRegex); err != nil {
		return err
	}

	return nil
}

func (t *reconcileTask) namespaceKeyReferences(fileFilter func(os.FileInfo, string) bool, createRegex func(string) *regexp.Regexp) error {
	keyRegexes := make([]*regexp.Regexp, len(t.keys))
	for i, key := range t.keys {
		keyRegexes[i] = createRegex(key)
	}

	fileChan, _ := java.ListFilesAsync(t.c.RootDir(), fileFilter)
	for f := range fileChan {
		if t.c.M.MainClass.ExcludeJavaFile(f.Info.Name()) {
			if t.pm.debug {
				log.Printf("[%s] Excluded from fixing conflicting property key: %s", t.c.Name, f.Info.Name())
			}
			continue
		}

		if err := t.namespaceKeyReferenceFile(f, keyRegexes); err != nil {
			return err
		}
	}

	return nil
}

func (t *reconcileTask) namespaceKeyReferenceFile(f java.FileInfo, keyRegexes []*regexp.Regexp) error {
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
				newKey := Key(t.keys[i]).WithNamespace(t.c.Name)
				replaced := t.replaceKeyInMatch(match, t.keys[i], newKey)
				dmp := diffmatchpatch.New()
				diffs := dmp.DiffMain(match, replaced, false)
				if t.pm.debug {
					log.Printf("[%s] Transforming %s\n%s", t.c.Name, f.Path, dmp.DiffPrettyText(diffs))
				}
				return replaced
			})

			if strings.Contains(regex.String(), "@ConfigurationProperties") {
				t.result.ConfigurationProperties++
			} else {
				t.result.KeyPrefixed++
			}
		}
	}

	if changed {
		// newContent 是叠加所有替换后的完整文件内容
		if err = ioutil.WriteFile(f.Path, []byte(newContent), f.Info.Mode()); err != nil {
			return err
		}
	}

	return nil
}

func (t *reconcileTask) replaceKeyInMatch(match, key, newKey string) string {
	if strings.Contains(match, "@ConfigurationProperties") {
		return strings.Replace(match, `"`+key+`"`, `"`+newKey+`"`, 1)
	}
	parts := strings.SplitN(match, "${", 2)
	if len(parts) == 2 {
		return parts[0] + "${" + newKey + strings.TrimPrefix(parts[1], key)
	}
	return match
}

func (t *reconcileTask) updateRequestMappings() error {
	contextPath := filepath.Clean("/" + strings.Trim(t.servletContextPath, "/"))
	ctx := context.WithValue(context.Background(), "contextPath", contextPath)
	return code.NewComponentJavaWalker(*t.c).
		AddVisitor(t).
		Walk(code.WithContext(ctx))
}

// 更新 @RequestMapping
func (t *reconcileTask) Visit(ctx context.Context, jf *code.JavaFile) {
	contextPath, ok := ctx.Value("contextPath").(string)
	if !ok {
		log.Fatalf("BUG: invalid contxt value: %+v", contextPath)
	}

	oldContent := jf.Content()
	newContent := t.updateRequestMappingInFile(oldContent, contextPath)
	if newContent != oldContent {
		ledger.Get().TransformRequestMapping(t.c.Name, "", contextPath)
		if t.pm.debug {
			diff.RenderUnifiedDiff(oldContent, newContent)
		}

		if err := jf.Overwrite(newContent); err != nil {
			log.Fatalf("%v", err)
		}
		t.result.RequestMapping++
	}
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
