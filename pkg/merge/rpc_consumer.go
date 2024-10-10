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
}

func NewRpcConsumerManager() *RpcConsumerManager {
	return &RpcConsumerManager{
		IntraComponentConflicts: make(map[string][]string),
		interfaceToComponent:    make(map[string]string),
		globalReferenceMap:      make(map[string]*etree.Element),
	}
}

func (dm *RpcConsumerManager) Reset() {
	dm.ScannedBeansCount = 0
	dm.IntraComponentConflicts = make(map[string][]string)
	dm.GeneratedBeansCount = 0
	dm.TargetFile = ""
	dm.interfaceToComponent = make(map[string]string)
	dm.globalReferenceMap = make(map[string]*etree.Element)
}

func (dm *RpcConsumerManager) MergeConsumerXmlFiles(m *manifest.Manifest, rpc string) error {
	if rpc == RpcDubbo && !m.DubboEnabled() {
		return nil
	}

	writeFile := false
	for _, component := range m.Components {
		componentConflicts := make(map[string]bool)

		var xmls []string
		switch rpc {
		case RpcJsf:
			xmls = component.JsfConsumerXmls
		case RpcDubbo:
			xmls = component.DubboConsumerXmls
		default:
			return fmt.Errorf("Unknown RPC type: %s", rpc)
		}

		if len(xmls) == 0 {
			log.Printf("[%s] Component[%s] skipped", rpc, component.Name)
			continue
		}

		writeFile = true
		for _, xmlFn := range xmls {
			filePath := filepath.Join(component.Name, xmlFn)
			if err := dm.processXmlFile(m, filePath, component, componentConflicts, rpc); err != nil {
				return err
			}
		}
	}

	if !writeFile {
		return nil
	}

	return dm.writeMergedXmlToFile(federated.GeneratedResourceBaseDir(m.Main.Name), rpc)
}

func (dm *RpcConsumerManager) processXmlFile(m *manifest.Manifest, filePath string, component manifest.ComponentInfo, componentConflicts map[string]bool, rpc string) error {
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
		log.Printf("[%s:%s] Processing %s import: %s", rpc, component.Name, filepath.Base(filePath), filepath.Base(importPath))

		if err := dm.processXmlFile(m, importPath, component, componentConflicts, rpc); err != nil {
			return err
		}
	}

	// Merge references from the current xml file
	dm.mergeReferences(doc.FindElements("//reference"), m, component, componentConflicts, rpc)

	return nil
}

func (dm *RpcConsumerManager) mergeReferences(references []*etree.Element, m *manifest.Manifest, component manifest.ComponentInfo, componentConflicts map[string]bool, rpc string) {
	for _, reference := range references {
		dm.ScannedBeansCount++
		interfaceName := reference.SelectAttrValue("interface", "")
		if m.Main.Reconcile.RpcConsumer.IgnoreInterface(interfaceName) {
			log.Printf("[%s:%s] Ignore rpc consumer: %s", rpc, component.Name, interfaceName)
			dm.IgnoredInterfaceN++
			continue
		}

		if _, exists := dm.globalReferenceMap[interfaceName]; !exists {
			dm.registerReference(component, reference, rpc)
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

func (dm *RpcConsumerManager) registerReference(component manifest.ComponentInfo, reference *etree.Element, rpc string) {
	interfaceName := reference.SelectAttrValue("interface", "")

	// 删除属性 scope="remote"，Apache Dubbo 才能自动切换到本地调用/injvm；JSF 没有该属性
	if scopeAttr := reference.SelectAttr("scope"); scopeAttr != nil && scopeAttr.Value == "remote" {
		reference.RemoveAttr("scope")
		log.Printf("[%s:%s] Removed scope=\"remote\" attr for: %s", rpc, component.Name, interfaceName)
	}

	dm.globalReferenceMap[interfaceName] = reference
	dm.interfaceToComponent[interfaceName] = component.Name
}

func (dm *RpcConsumerManager) writeMergedXmlToFile(targetDir string, rpc string) error {
	dm.GeneratedBeansCount = len(dm.globalReferenceMap)

	doc := etree.NewDocument()
	beans := doc.CreateElement("beans")
	beans.CreateAttr("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
	beans.CreateAttr("xmlns", "http://www.springframework.org/schema/beans")
	var targetFile string
	switch rpc {
	case RpcJsf:
		targetFile = federatedJSFConsumerXmlFn
		beans.CreateAttr("xmlns:jsf", "http://jsf.jd.com/schema/jsf")
		beans.CreateAttr("xsi:schemaLocation", "http://www.springframework.org/schema/beans http://www.springframework.org/schema/beans/spring-beans.xsd\n       http://jsf.jd.com/schema/jsf http://jsf.jd.com/schema/jsf/jsf.xsd")
	case RpcDubbo:
		targetFile = federatedDubboConsumerXmlFn
		beans.CreateAttr("xmlns:dubbo", "http://dubbo.apache.org/schema/dubbo")
		beans.CreateAttr("xsi:schemaLocation", "http://www.springframework.org/schema/beans http://www.springframework.org/schema/beans/spring-beans.xsd\n       http://dubbo.apache.org/schema/dubbo http://dubbo.apache.org/schema/dubbo/dubbo.xsd")
	default:
		return fmt.Errorf("Unknown RPC type: %s", rpc)
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
