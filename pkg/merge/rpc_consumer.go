package merge

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"federate/pkg/federated"
	"federate/pkg/manifest"
	"github.com/beevik/etree"
)

type RpcConsumerManager struct {
	ScannedBeansCount   int
	GeneratedBeansCount int
	IgnoredInterfaceN   int

	IntraComponentConflicts map[string][]string
	InterComponentConflicts []string
	TargetFile              string

	interfaceToComponent map[string]string
	globalReferenceMap   map[string]*etree.Element

	rpcType string
}

func NewRpcConsumerManager(rpcType string) *RpcConsumerManager {
	return &RpcConsumerManager{
		IntraComponentConflicts: make(map[string][]string),
		interfaceToComponent:    make(map[string]string),
		globalReferenceMap:      make(map[string]*etree.Element),
		rpcType:                 rpcType,
	}
}

func (dm *RpcConsumerManager) referenceXmlTag() (tag string) {
	switch dm.rpcType {
	case RpcJsf: // <jsf:consumer>
		tag = "//consumer"
	case RpcDubbo: // <dubbo:reference>
		tag = "//reference"
	}
	return
}

func (dm *RpcConsumerManager) RPC() string {
	return dm.rpcType
}

func (dm *RpcConsumerManager) Reset() {
	dm.ScannedBeansCount = 0
	dm.IntraComponentConflicts = make(map[string][]string)
	dm.GeneratedBeansCount = 0
	dm.TargetFile = ""
	dm.interfaceToComponent = make(map[string]string)
	dm.globalReferenceMap = make(map[string]*etree.Element)
}

func (dm *RpcConsumerManager) MergeConsumerXmlFiles(m *manifest.Manifest) error {
	if dm.rpcType == RpcDubbo && !m.DubboEnabled() {
		return nil
	}

	writeFile := false
	for _, component := range m.Components {
		componentConflicts := make(map[string]bool)

		var xmlPatterns []string
		switch dm.rpcType {
		case RpcJsf:
			xmlPatterns = component.Resources.JsfConsumerXmls
		case RpcDubbo:
			xmlPatterns = component.Resources.DubboConsumerXmls
		default:
			return fmt.Errorf("Unknown RPC type: %s", dm.rpcType)
		}

		if len(xmlPatterns) == 0 {
			log.Printf("[%s] Component[%s] skipped", dm.rpcType, component.Name)
			continue
		}

		writeFile = true
		for _, xmlPattern := range xmlPatterns {
			for _, baseDir := range component.Resources.BaseDirs {
				sourceDir := filepath.Join(component.Name, baseDir)
				fullPattern := filepath.Join(sourceDir, xmlPattern)

				// 使用 filepath.Glob 来匹配文件
				matches, err := filepath.Glob(fullPattern)
				if err != nil {
					return fmt.Errorf("error matching pattern %s: %v", fullPattern, err)
				}

				if len(matches) == 0 {
					continue
				}

				for _, filePath := range matches {
					log.Printf("[%s:%s] Processing %s", dm.rpcType, component.Name, filepath.Base(filePath))
					if err := dm.processXmlFile(m, filePath, component, componentConflicts); err != nil {
						return fmt.Errorf("error processing file %s: %v", filePath, err)
					}
				}
			}
		}
	}

	if !writeFile {
		return nil
	}

	return dm.writeMergedXmlToFile(federated.GeneratedResourceBaseDir(m.Main.Name))
}

func (dm *RpcConsumerManager) processXmlFile(m *manifest.Manifest, filePath string, component manifest.ComponentInfo, componentConflicts map[string]bool) error {
	doc := etree.NewDocument()
	if err := doc.ReadFromFile(filePath); err != nil {
		return err
	}

	if err := m.State.AddMergeSource(filePath, component); err != nil {
		return err
	}

	// Process import resources recursively
	for _, importElem := range doc.FindElements("//import") {
		resourceAttr := importElem.SelectAttrValue("resource", "")
		// 移除 resource 属性中的 classpath: 前缀
		resourceAttr = strings.TrimPrefix(resourceAttr, "classpath:")
		importPath := filepath.Join(filepath.Dir(filePath), resourceAttr)
		log.Printf("[%s:%s] Processing %s import: %s", dm.rpcType, component.Name, filepath.Base(filePath), filepath.Base(importPath))

		if err := dm.processXmlFile(m, importPath, component, componentConflicts); err != nil {
			return err
		}
	}

	// Merge references from the current xml file
	dm.mergeReferences(doc.FindElements(dm.referenceXmlTag()), m, component, componentConflicts)

	return nil
}

func (dm *RpcConsumerManager) mergeReferences(references []*etree.Element, m *manifest.Manifest, component manifest.ComponentInfo, componentConflicts map[string]bool) {
	for _, reference := range references {
		dm.ScannedBeansCount++
		interfaceName := reference.SelectAttrValue("interface", "")
		if interfaceName == "" {
			continue
		}
		if m.Main.Reconcile.Rpc.Consumer.IgnoreInterface(interfaceName) {
			log.Printf("[%s:%s] Ignore rpc consumer: %s", dm.rpcType, component.Name, interfaceName)
			dm.IgnoredInterfaceN++
			continue
		}

		if _, exists := dm.globalReferenceMap[interfaceName]; !exists {
			dm.registerReference(component, reference)
			continue
		}

		existingComponent := dm.interfaceToComponent[interfaceName]
		if existingComponent == component.Name {
			componentConflicts[interfaceName] = true
			dm.IntraComponentConflicts[component.Name] = append(dm.IntraComponentConflicts[component.Name], interfaceName)
		} else {
			dm.InterComponentConflicts = append(dm.InterComponentConflicts, interfaceName)
		}
	}
}

func (dm *RpcConsumerManager) registerReference(component manifest.ComponentInfo, reference *etree.Element) {
	interfaceName := reference.SelectAttrValue("interface", "")

	// 删除属性 scope="remote"，Apache Dubbo 才能自动切换到本地调用/injvm；JSF 没有该属性
	if scopeAttr := reference.SelectAttr("scope"); scopeAttr != nil && scopeAttr.Value == "remote" {
		reference.RemoveAttr("scope")
		log.Printf("[%s:%s] Removed scope=\"remote\" attr for: %s", dm.rpcType, component.Name, interfaceName)
	}

	dm.globalReferenceMap[interfaceName] = reference
	dm.interfaceToComponent[interfaceName] = component.Name
}

func (dm *RpcConsumerManager) writeMergedXmlToFile(targetDir string) error {
	dm.GeneratedBeansCount = len(dm.globalReferenceMap)

	doc := etree.NewDocument()
	beans := doc.CreateElement("beans")
	beans.CreateAttr("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
	beans.CreateAttr("xmlns", "http://www.springframework.org/schema/beans")
	var targetFile string
	switch dm.rpcType {
	case RpcJsf:
		targetFile = federatedJSFConsumerXmlFn
		beans.CreateAttr("xmlns:jsf", "http://jsf.jd.com/schema/jsf")
		beans.CreateAttr("xsi:schemaLocation", "http://www.springframework.org/schema/beans http://www.springframework.org/schema/beans/spring-beans.xsd\n       http://jsf.jd.com/schema/jsf http://jsf.jd.com/schema/jsf/jsf.xsd")
	case RpcDubbo:
		targetFile = federatedDubboConsumerXmlFn
		beans.CreateAttr("xmlns:dubbo", "http://dubbo.apache.org/schema/dubbo")
		beans.CreateAttr("xsi:schemaLocation", "http://www.springframework.org/schema/beans http://www.springframework.org/schema/beans/spring-beans.xsd\n       http://dubbo.apache.org/schema/dubbo http://dubbo.apache.org/schema/dubbo/dubbo.xsd")
	default:
		return fmt.Errorf("Unknown RPC type: %s", dm.rpcType)
	}

	for _, reference := range dm.globalReferenceMap {
		beans.AddChild(reference)
	}

	dm.TargetFile = filepath.Join(targetDir, targetFile)
	if err := os.MkdirAll(filepath.Dir(dm.TargetFile), 0755); err != nil {
		return fmt.Errorf("error creating directories for %s: %v", dm.TargetFile, err)
	}

	// Set custom indent for better formatting
	doc.Indent(4)

	// Write the document to file
	if err := doc.WriteToFile(dm.TargetFile); err != nil {
		return fmt.Errorf("error writing to file %s: %v", dm.TargetFile, err)
	}

	return nil
}
