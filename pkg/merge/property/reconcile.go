package property

import (
	"log"
	"os"
	"runtime"

	"federate/pkg/concurrent"
)

// 根据扫描的冲突情况进行调和，处理 .yml & .properties
func (cm *PropertyManager) Reconcile(dryRun bool) (err error) {
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

			if cm.debug {
				log.Printf("[%s] fixing conflict key=%s value=%v", componentName, key, value)
			}
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
			dryRun:             dryRun,
			servletContextPath: cm.servletContextPath[componentName],
			result:             ReconcileReport{},
		})
	}

	errors := executor.Execute()
	if len(errors) > 0 {
		err = errors[0] // 返回第一个遇到的错误
	}

	// aggregate resport
	for _, task := range executor.Tasks() {
		reconcileTask := task.(*reconcileTask)
		cm.result.KeyPrefixed += reconcileTask.result.KeyPrefixed
		cm.result.RequestMapping += reconcileTask.result.RequestMapping
		cm.result.ConfigurationProperties += reconcileTask.result.ConfigurationProperties
	}

	return
}

func (cm *PropertyManager) namespacePropertyPlaceholders(s, componentName string) string {
	return os.Expand(s, func(key string) string {
		return "${" + componentName + "." + key + "}"
	})
}
