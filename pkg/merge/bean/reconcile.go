package bean

import (
	"fmt"
	"log"

	"federate/pkg/diff"
	"federate/pkg/federated"
	"federate/pkg/java"
	"federate/pkg/manifest"
	"federate/pkg/util"
	"github.com/beevik/etree"
)

// 解决 bean id 冲突：此时资源文件都已经拷贝到目标目录
func (b *XmlBeanManager) Reconcile() error {
	b.loadBeans()
	b.makeReconcilePlan()

	if b.plan.HasConflict() {
		b.plan.ShowConflictReport()
	}

	b.executeReconcilePlan()
	b.showRisk()
	return nil
}

func (b *XmlBeanManager) executeReconcilePlan() {
	log.Printf("Executing reconcile plan ...")

	// 修改 bean id
	for xmlPath, modificationPlan := range b.plan.beanIdModificationFiles {
		err := b.updateBeanIdsInFile(xmlPath, modificationPlan)
		if err != nil {
			log.Fatalf("Error updating bean ids in file %s: %v", xmlPath, err)
		}
	}

	// 删除多余的 bean class
	for _, component := range b.m.Components {
		if err := b.removeRedundantBeanClassesInDir(component); err != nil {
			log.Fatalf("Error removing redundant beans: %v", err)
		}
	}

	// 更新引用
	for _, component := range b.m.Components {
		if err := b.updateBeanRefsInDir(component.TargetResourceDir(), component.Name); err != nil {
			log.Fatalf("Error fixing bean refs: %v", err)
		}
	}
}

func (b *XmlBeanManager) removeRedundantBeanClassesInDir(component manifest.ComponentInfo) error {
	files, err := java.ListXMLFiles(component.TargetResourceDir())
	if err != nil {
		return err
	}

	for _, path := range files {
		if err := b.removeRedundantBeanClassesInFile(path, component); err != nil {
			return err
		}
	}
	return nil
}

func (b *XmlBeanManager) removeRedundantBeanClassesInFile(filePath string, component manifest.ComponentInfo) error {
	doc := etree.NewDocument()
	if err := doc.ReadFromFile(filePath); err != nil {
		return err
	}

	root := doc.Root()

	// Find all bean elements across all namespaces
	beans := root.FindElements(".//*[local-name()='bean']")

	update := false
	for _, beanElement := range beans {
		classFullName := b.extractClassFullName(beanElement)
		if b.plan.IsRedundantClass(classFullName, filePath) {
			log.Printf("[%s] Killing %s from %s", component.Name, java.ClassSimpleName(classFullName), federated.ResourceBaseName(filePath))
			update = true
			parent := beanElement.Parent()
			if parent != nil {
				parent.RemoveChild(beanElement)
			}
		}
	}

	// Save the modified XML
	if update {
		return doc.WriteToFile(filePath)
	}

	return nil
}

func (b *XmlBeanManager) updateBeanIdsInFile(filePath string, modificationPlan map[string]string) error {
	if len(modificationPlan) < 1 {
		return fmt.Errorf("Empty plan: %v", modificationPlan)
	}

	doc := etree.NewDocument()
	if err := doc.ReadFromFile(filePath); err != nil {
		return err
	}

	root := doc.Root()
	oldXml, err := doc.WriteToString()
	if err != nil {
		return err
	}

	modifiedCount := b.updateBeanIdsInElement(root, modificationPlan, filePath)
	if modifiedCount != len(modificationPlan) {
		return fmt.Errorf("Expected %d, actual %d updates on %s", len(modificationPlan), modifiedCount, filePath)
	}

	newXml, err := doc.WriteToString()
	if err != nil {
		return err
	}

	log.Printf("Rewritten %d/%d bean id: %s", modifiedCount, len(modificationPlan), filePath)
	diff.RenderUnifiedDiff(oldXml, newXml)
	return doc.WriteToFile(filePath)
}

func (b *XmlBeanManager) updateBeanIdsInElement(element *etree.Element, modificationPlan map[string]string, filePath string) int {
	modifiedCount := 0

	// 处理所有具有 'id' 属性的元素，而不仅仅是 'bean' 元素
	// <jsf:consumer id="foo">
	// <jsf:consumerGroup id="foo">
	// <dubbo:reference id="foo">
	beans := element.FindElements(".//*[@id]")
	for _, bean := range beans {
		if beanId := bean.SelectAttrValue("id", ""); beanId != "" {
			if newId, ok := modificationPlan[beanId]; ok {
				util.UpdateXmlElement(bean, "id", newId)
				modifiedCount++
			}
		}
	}

	// 递归处理子元素
	for _, child := range element.ChildElements() {
		modifiedCount += b.updateBeanIdsInElement(child, modificationPlan, filePath)
	}
	return modifiedCount
}

// updateBeanRefsInDir 更新目录中的 bean 引用
func (b *XmlBeanManager) updateBeanRefsInDir(dir, componentName string) error {
	files, err := java.ListXMLFiles(dir)
	if err != nil {
		return err
	}

	for _, path := range files {
		if modificationPlan, exists := b.plan.beanIdModificationFiles[path]; exists {
			if err := b.updateBeanRefsInFile(path, modificationPlan, componentName); err != nil {
				return err
			}
		}
	}
	return nil
}

func (b *XmlBeanManager) updateBeanRefsInFile(filePath string, modificationPlan map[string]string, componentName string) error {
	doc := etree.NewDocument()
	if err := doc.ReadFromFile(filePath); err != nil {
		return err
	}

	root := doc.Root()
	oldXml, err := doc.WriteToString()
	if err != nil {
		return err
	}

	modifiedCount := b.updateBeanRefsInElement(root, modificationPlan, filePath)
	if modifiedCount > 0 {
		newXml, err := doc.WriteToString()
		if err != nil {
			return err
		}

		log.Printf("Rewritten %d bean ref: %s", modifiedCount, filePath)
		diff.RenderUnifiedDiff(oldXml, newXml)
		return doc.WriteToFile(filePath)
	}
	return nil
}

func (b *XmlBeanManager) updateBeanRefsInElement(element *etree.Element, modificationPlan map[string]string, xmlFile string) int {
	modifiedCount := 0

	// 定义需要更新的属性列表
	refAttributes := []string{"ref", "value-ref", "bean", "properties-ref"}
	for _, attr := range refAttributes {
		if ref := element.SelectAttrValue(attr, ""); ref != "" {
			if newRef, ok := modificationPlan[ref]; ok {
				util.UpdateXmlElement(element, attr, newRef)
				modifiedCount++
			}
		}
	}

	// 递归处理子元素，例如：list/map 等内部的 ref 值修改
	for _, child := range element.ChildElements() {
		modifiedCount += b.updateBeanRefsInElement(child, modificationPlan, xmlFile)
	}
	return modifiedCount
}
