package merge

import (
	"fmt"
	"io/ioutil"
	"log"
	"path/filepath"
	"strings"

	"federate/pkg/code"
	"federate/pkg/java"
	"federate/pkg/manifest"
	"federate/pkg/merge/property"
	"federate/pkg/merge/transformer"
	"github.com/beevik/etree"
)

// Java源代码和XML
type envManager struct {
	m *manifest.Manifest
	p *property.PropertyManager
}

func NewEnvManager(m *manifest.Manifest, propManager *property.PropertyManager) Reconciler {
	return newEnvManager(m, propManager)
}

func newEnvManager(m *manifest.Manifest, propManager *property.PropertyManager) *envManager {
	return &envManager{m: m, p: propManager}
}

// TODO xml 也可能引用环境变量
func (e *envManager) Reconcile(dryRun bool) error {
	envKeys := make(map[string]struct{})

	for _, component := range e.m.Components {
		// java 源代码里的环境变量引用
		paths, err := java.ListJavaMainSourceFiles(component.RootDir())
		if err != nil {
			log.Printf("Error walking the path %s: %v", component.RootDir(), err)
			continue
		}

		for _, path := range paths {
			keys, err := e.findEnvRefsInJava(path)
			if err != nil {
				log.Printf("Error processing file %s: %v", path, err)
				return nil
			}
			for _, key := range keys {
				envKeys[key] = struct{}{}
			}
		}

		// XML 里环境变量引用
		for _, root := range component.ResourceBaseDirs() {
			xmlPaths, err := java.ListXMLFiles(root)
			if err != nil {
				log.Printf("Error listing XML files in %s: %v", root, err)
				continue
			}

			for _, xmlPath := range xmlPaths {
				keys, err := e.findEnvRefsInXML(component, xmlPath)
				if err != nil {
					log.Printf("Error processings %s: %v", xmlPath, err)
					continue
				}

				for _, key := range keys {
					envKeys[key] = struct{}{}
				}
			}
		}
	}

	if len(envKeys) > 0 {
		for key := range envKeys {
			transformer.Get().RegisterEnvKey(key)
		}
	} else {
		log.Println("System.getProperty is OK")
	}
	return nil
}

func (e *envManager) findEnvRefsInJava(filePath string) ([]string, error) {
	content, err := ioutil.ReadFile(filePath)
	if err != nil {
		return nil, err
	}

	matches := code.P.SystemGetPropertyRegex.FindAllSubmatch(content, -1)

	keys := make([]string, 0, len(matches))
	for _, match := range matches {
		if len(match) > 1 {
			arg := strings.TrimSpace(string(match[1]))
			if strings.HasPrefix(arg, "\"") && strings.HasSuffix(arg, "\"") {
				// 直接的字符串字面量
				keys = append(keys, arg[1:len(arg)-1])
			} else if strings.HasPrefix(arg, "'") && strings.HasSuffix(arg, "'") {
				// 使用单引号的字符串字面量
				keys = append(keys, arg[1:len(arg)-1])
			} else {
				// 可能是变量或常量
				keys = append(keys, "VARIABLE: "+arg)
			}
		}
	}

	return keys, nil
}

func (e *envManager) findEnvRefsInXML(c manifest.ComponentInfo, filePath string) ([]string, error) {
	var keys []string
	doc := etree.NewDocument()
	if err := doc.ReadFromFile(filePath); err != nil {
		return nil, fmt.Errorf("error reading XML file %s: %v", filePath, err)
	}

	root := doc.Root()
	if root == nil {
		return nil, fmt.Errorf("empty XML file %s", filePath)
	}

	// 遍历所有元素和属性
	e.searchElementForEnvRefs(c, filePath, root, &keys)

	return keys, nil
}

func (e *envManager) searchElementForEnvRefs(c manifest.ComponentInfo, filePath string, elem *etree.Element, keys *[]string) {
	// 检查元素的属性
	for _, attr := range elem.Attr {
		matches := code.P.XmlEnvRef.FindAllStringSubmatch(attr.Value, -1)
		for _, match := range matches {
			if len(match) > 1 {
				*keys = append(*keys, match[1])
			}
		}
	}

	// 检查元素的文本内容
	if elem.Text() != "" {
		matches := code.P.XmlEnvRef.FindAllStringSubmatch(elem.Text(), -1)
		for _, match := range matches {
            // 属性管理器里没有定义的才是环境变量
			if len(match) > 1 && (e.p == nil || !e.p.ContainsKey(c, match[1])) {
				*keys = append(*keys, match[1])
			}
		}
	}

	// 递归检查子元素
	for _, child := range elem.ChildElements() {
		e.searchElementForEnvRefs(c, filePath, child, keys)
	}
}
