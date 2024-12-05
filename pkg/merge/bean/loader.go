package bean

import (
	"log"
	"path/filepath"

	"federate/pkg/java"
	"federate/pkg/manifest"
	"github.com/beevik/etree"
)

// loadBeans load xml beans from component src dir into memory
func (b *XmlBeanManager) loadBeans() error {
	for _, component := range b.m.Components {
		dir := component.TargetResourceDir()
		fileChan, _ := java.ListXMLFilesAsync(dir)
		for f := range fileChan {
			if !b.m.IgnoreResourceSrcFile(f.Info, component) {
				doc := etree.NewDocument()
				if err := doc.ReadFromFile(f.Path); err != nil {
					return err
				}

				if err := b.processBeans(doc.Root(), component, dir, f.Path, []string{}, ""); err != nil {
					return err
				}
			}
		}
	}
	log.Printf("XML Beans Detected: %d", len(b.beanIdMap))
	return nil
}

func (b *XmlBeanManager) processBeans(element *etree.Element, component manifest.ComponentInfo, sourceDir, path string,
	parentPath []string, parentId string) error {
	id := element.SelectAttrValue("id", "")
	class := b.extractClassFullName(element)
	if id != "" {
		fullId := id
		if parentId != "" {
			fullId = parentId + beanIdPathSeparator + id
		}
		relativePath, err := filepath.Rel(sourceDir, path)
		if err != nil {
			return err
		}
		targetFilePath := filepath.Join(component.TargetResourceDir(), relativePath)
		b.registerBean(fullId, component.Name, path, targetFilePath, parentPath, class)
	}

	// 递归处理子元素
	for _, child := range element.ChildElements() {
		newParentPath := append(parentPath, element.Tag)
		newParentId := parentId
		if id != "" {
			if newParentId != "" {
				newParentId += beanIdPathSeparator
			}
			newParentId += id
		}
		if err := b.processBeans(child, component, sourceDir, path, newParentPath, newParentId); err != nil {
			return err
		}
	}
	return nil
}

// registerBean 添加一个 bean id 到管理器
func (b *XmlBeanManager) registerBean(beanId, componentName, sourceFilePath, targetFilePath string, parentPath []string, classFullName string) {
	b.beanIdMap[beanId] = append(b.beanIdMap[beanId], BeanIdInfo{
		BeanId:         beanId,
		ComponentName:  componentName,
		SourceFilePath: sourceFilePath,
		TargetFilePath: targetFilePath,
		ParentPath:     parentPath,
		ClassFullName:  classFullName,
	})
}

// extractClassFullName extracts the full class name from various Spring XML attributes and tags
func (b *XmlBeanManager) extractClassFullName(element *etree.Element) string {
	// Check for 'class' attribute
	if class := element.SelectAttrValue("class", ""); class != "" {
		return class
	}

	// Check for 'interface' attribute
	if inf := element.SelectAttrValue("interface", ""); inf != "" {
		return inf
	}

	// Check for 'value-type' attribute
	if valueType := element.SelectAttrValue("value-type", ""); valueType != "" {
		return valueType
	}

	// Check for 'factory-bean' and 'factory-method' combination
	if factoryBean := element.SelectAttrValue("factory-bean", ""); factoryBean != "" {
		if factoryMethod := element.SelectAttrValue("factory-method", ""); factoryMethod != "" {
			return factoryBean + "." + factoryMethod
		}
	}

	// Check for nested 'bean' element with 'class' attribute
	if nestedBean := element.SelectElement("bean"); nestedBean != nil {
		if nestedClass := nestedBean.SelectAttrValue("class", ""); nestedClass != "" {
			return nestedClass
		}
	}

	return ""
}
