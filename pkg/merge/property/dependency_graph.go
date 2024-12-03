package property

import (
	"strings"

	"federate/pkg/primitive"
)

// PropertyNode 代表依赖图中的一个属性节点
type PropertyNode struct {
	Key      string
	Value    string
	Resolved bool
	Deps     *primitive.StringSet // 存储该节点所依赖的其他节点
}

// DependencyGraph 维护整个属性系统的依赖关系
type DependencyGraph struct {
	Nodes map[string]*PropertyNode
}

func NewDependencyGraph() *DependencyGraph {
	return &DependencyGraph{
		Nodes: make(map[string]*PropertyNode),
	}
}

func (dg *DependencyGraph) AddNode(key, value string) {
	node := &PropertyNode{
		Key:   key,
		Value: value,
		Deps:  primitive.NewStringSet(),
	}
	dg.Nodes[key] = node
}

func (dg *DependencyGraph) AddDependency(from, to string) {
	if node, exists := dg.Nodes[from]; exists {
		node.Deps.Add(to)
	}
}

// TopologicalSort 执行拓扑排序并返回排序后的节点键列表.
// 返回的切片包含所有节点的键,按照依赖关系排序:
// 如果节点A依赖节点B,则B将在结果中出现在A之前.
// 注意:如果存在循环依赖,函数会跳过循环的部分,但不会报错.
func (dg *DependencyGraph) TopologicalSort() []string {
	var sorted []string
	visited := primitive.NewStringSet()
	cycleDetector := primitive.NewStringSet() // 用于检测循环依赖

	var visit func(string)
	visit = func(key string) {
		if cycleDetector.Contains(key) {
			// 检测到循环依赖，直接返回
			return
		}
		if !visited.Contains(key) {
			cycleDetector.Add(key)
			node, exists := dg.Nodes[key]
			if exists {
				for dep := range node.Deps.Items() {
					visit(dep) // 递归访问依赖节点
				}
			}
			visited.Add(key)
			cycleDetector.Remove(key)
			sorted = append(sorted, key) // 将节点添加到排序结果
		}
	}

	for key := range dg.Nodes {
		if !visited.Contains(key) {
			visit(key)
		}
	}

	return sorted
}

func (dg *DependencyGraph) FromPropertyManager(pm *PropertyManager) *DependencyGraph {
	if pm == nil || pm.r == nil {
		return dg
	}

	for component, entries := range pm.r.GetAllResolvableEntries() {
		for key, entry := range entries {
			fullKey := component + "." + key
			dg.AddNode(fullKey, entry.StringValue())

			// 解析占位符以找到依赖
			matches := P.placeholderRegex.FindAllStringSubmatch(entry.StringValue(), -1)
			for _, match := range matches {
				if len(match) > 1 {
					depKey := match[1]
					if !strings.Contains(depKey, ":") { // 忽略默认值
						if !strings.Contains(depKey, ".") {
							// 如果依赖键不包含点，则为本地引用
							depKey = component + "." + depKey
						}
						dg.AddNode(depKey, "") // 确保依赖节点存在
						dg.AddDependency(fullKey, depKey)
					}
				}
			}
		}
	}

	return dg
}
