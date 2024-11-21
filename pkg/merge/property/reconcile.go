package property

import (
	"os"
	"runtime"

	"federate/pkg/concurrent"
)

type ReconcileReport struct {
	KeyPrefixed             int
	RequestMapping          int
	ConfigurationProperties int
}

// 根据扫描的冲突情况进行调和，处理 .yml & .properties
func (cm *PropertyManager) Reconcile(dryRun bool) (err error) {
	// pass 1: 识别冲突
	conflicts := cm.identifyAllConflicts()
	if len(conflicts) == 0 {
		return
	}

	// pass 2:
	conflictingKeysOfComponents := make(map[string][]string) // Group keys by component
	for key, components := range conflicts {
		for componentName, value := range components {
			conflictingKeysOfComponents[componentName] = append(conflictingKeysOfComponents[componentName], key)

			cm.r.NamespaceProperty(componentName, Key(key), value)
		}
	}

	// pass 3: 创建并发任务对 Component 源代码进行插桩改写
	executor := concurrent.NewParallelExecutor(runtime.NumCPU())
	executor.SetName("Overwrite Java/XML conflicted property references & @RequestMapping & @ConfigurationProperties")
	for componentName, keys := range conflictingKeysOfComponents {
		executor.AddTask(&reconcileTask{
			cm:                 cm,
			component:          cm.m.ComponentByName(componentName),
			keys:               keys,
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
		cm.result.KeyPrefixed += reconcileTask.result.keyPrefixed
		cm.result.RequestMapping += reconcileTask.result.requestMapping
		cm.result.ConfigurationProperties += reconcileTask.result.configurationProperties
	}

	return
}

func (cm *PropertyManager) namespacePropertyPlaceholders(s, componentName string) string {
	return os.Expand(s, func(key string) string {
		return "${" + componentName + "." + key + "}"
	})
}
