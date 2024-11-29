package merge

import (
	"context"
	"fmt"
	"log"
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

func NewEnvManager(propManager *property.PropertyManager) Reconciler {
	return newEnvManager(propManager.M(), propManager)
}

func newEnvManager(m *manifest.Manifest, propManager *property.PropertyManager) *envManager {
	return &envManager{m: m, p: propManager}
}

func (e *envManager) Name() string {
	return "Reconciling ENV variables conflicts"
}

func (e *envManager) Reconcile() error {
	// java 源代码里的环境变量引用
	for _, component := range e.m.Components {
		code.NewComponentJavaWalker(component).
			AddVisitor(e).
			Walk()
	}

	// XML 里环境变量引用
	for _, component := range e.m.Components {
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
					transformer.Get().RegisterEnvKey(component.Name, key)
				}
			}
		}
	}

	return nil
}

func (e *envManager) Visit(ctx context.Context, jf *code.JavaFile) {
	keys, err := e.findEnvRefsInJava(jf)
	if err != nil {
		log.Fatalf("%v", err)
	}

	for _, key := range keys {
		transformer.Get().RegisterEnvKey(jf.ComponentName(), key)
	}
}

func (e *envManager) findEnvRefsInJava(jf *code.JavaFile) ([]string, error) {
	matches := code.P.SystemGetPropertyRegex.FindAllSubmatch(jf.Bytes(), -1)

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
				keys = append(keys, arg)
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
				key := match[1]
				if strings.ToUpper(key) == key {
					// 只有全大写才认为是环境变量，否则没有办法识别
					*keys = append(*keys, key)
				}
			}
		}
	}

	// 检查元素的文本内容
	if elem.Text() != "" {
		matches := code.P.XmlEnvRef.FindAllStringSubmatch(elem.Text(), -1)
		for _, match := range matches {
			// 属性管理器里没有定义的才是环境变量
			if len(match) > 1 && (e.p == nil || !e.p.ContainsKey(c, match[1])) {
				key := match[1]
				if strings.ToUpper(key) == key {
					*keys = append(*keys, key)
				}
			}
		}
	}

	// 递归检查子元素
	for _, child := range elem.ChildElements() {
		e.searchElementForEnvRefs(c, filePath, child, keys)
	}
}
