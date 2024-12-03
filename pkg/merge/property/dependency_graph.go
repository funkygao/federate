package property

import (
	"strings"
)

type PropertyNode struct {
	Key      string
	Value    string
	Resolved bool
	Deps     map[string]struct{}
}

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
		Deps:  make(map[string]struct{}),
	}
	dg.Nodes[key] = node
}

func (dg *DependencyGraph) AddDependency(from, to string) {
	if node, exists := dg.Nodes[from]; exists {
		node.Deps[to] = struct{}{}
	}
}

func (dg *DependencyGraph) TopologicalSort() []string {
	var sorted []string
	visited := make(map[string]bool)
	temp := make(map[string]bool)

	var visit func(string)
	visit = func(key string) {
		if temp[key] {
			// Cyclic dependency detected
			return
		}
		if !visited[key] {
			temp[key] = true
			node, exists := dg.Nodes[key]
			if exists {
				for dep := range node.Deps {
					visit(dep)
				}
			}
			visited[key] = true
			temp[key] = false
			sorted = append(sorted, key)
		}
	}

	for key := range dg.Nodes {
		if !visited[key] {
			visit(key)
		}
	}

	// 不需要反转排序结果
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

			matches := P.placeholderRegex.FindAllStringSubmatch(entry.StringValue(), -1)
			for _, match := range matches {
				if len(match) > 1 {
					depKey := match[1]
					if !strings.Contains(depKey, ":") { // Ignore default values
						if !strings.Contains(depKey, ".") {
							// If the dependency key doesn't contain a dot, it's a local reference
							depKey = component + "." + depKey
						}
						dg.AddNode(depKey, "") // Ensure the dependency node exists
						dg.AddDependency(fullKey, depKey)
					}
				}
			}
		}
	}

	return dg
}
