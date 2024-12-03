package property

import (
	"log"
	"path/filepath"
	"runtime"

	"federate/pkg/concurrent"
	"federate/pkg/federated"
)

func (cm *PropertyManager) Name() string {
	return "Reconciling Property Conflicts References by Rewriting @Value/@ConfigurationProperties/@RequestMapping"
}

// 根据扫描的冲突情况进行调和，处理 .yml & .properties
func (cm *PropertyManager) Reconcile() (err error) {
	if err = cm.Prepare(); err != nil {
		return
	}

	// pass 1: 识别冲突
	conflicts := cm.identifyAllConflicts()
	if len(conflicts) == 0 {
		return
	}

	// pass 2: 注册表调和冲突，为冲突key增加前缀ns，对于 ConfigurationProperties/integralKey 整体处理
	conflictingKeysOfComponents := make(map[string][]string) // Group keys by component
	for key, components := range conflicts {
		for componentName, value := range components {
			conflictingKeysOfComponents[componentName] = append(conflictingKeysOfComponents[componentName], key)

			cm.r.SegregateProperty(componentName, Key(key), value)
		}
	}

	// pass 3: 创建并发任务对 Component 源代码进行插桩改写
	executor := concurrent.NewParallelExecutor(runtime.NumCPU())
	executor.SetName("Overwrite Java/XML conflicted property references & @RequestMapping & @ConfigurationProperties")
	for componentName, keys := range conflictingKeysOfComponents {
		executor.AddTask(&reconcileTask{
			c:                  cm.m.ComponentByName(componentName),
			keys:               keys,
			servletContextPath: cm.servletContextPath[componentName],
			result:             ReconcileReport{},
		})
	}

	errors := executor.Execute()
	if len(errors) > 0 {
		err = errors[0]
		return
	}

	// pass 4: 合并到目标文件
	if cm.writeTarget {
		if err = cm.writeTargetFiles(); err != nil {
			return
		}
	}

	// aggregate resport
	for _, task := range executor.Tasks() {
		reconcileTask := task.(*reconcileTask)
		cm.result.KeyPrefixed += reconcileTask.result.KeyPrefixed
		cm.result.RequestMapping += reconcileTask.result.RequestMapping
		cm.result.ConfigurationProperties += reconcileTask.result.ConfigurationProperties
	}

	log.Printf("Source code rewritten, @RequestMapping: %d, @Value: %d, @ConfigurationProperties: %d",
		cm.result.RequestMapping, cm.result.KeyPrefixed, cm.result.ConfigurationProperties)
	return
}

func (cm *PropertyManager) writeTargetFiles() (err error) {
	pn := filepath.Join(federated.GeneratedResourceBaseDir(cm.m.Main.Name), "application.properties")
	if err = cm.generateMergedPropertiesFile(pn); err != nil {
		return err
	}

	an := filepath.Join(federated.GeneratedResourceBaseDir(cm.m.Main.Name), "application.yml")
	if err = cm.generateMergedYamlFile(an); err != nil {
		return err
	}

	return
}
