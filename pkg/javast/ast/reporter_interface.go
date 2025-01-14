package ast

import (
	"fmt"
	"log"
	"sort"

	"federate/pkg/tabular"
)

func (i *Info) showInterfacesReport() {
	interfaceCounts := make(map[string]int)
	for _, interfaces := range i.Interfaces {
		for _, iface := range interfaces {
			interfaceCounts[iface]++
		}
	}

	i.showNameCountSection("Implemented Interfaces", []string{"Interface"}, topN(mapToNameCount(interfaceCounts), TopK))

	// 构建接口树并找出显著的集群
	tree := i.buildInterfaceTree()
	roots := findRootInterfaceClusters(tree)

	significantClusters := []InterfaceCluster{}
	for _, root := range roots {
		// 只考虑接口节点
		if isInterface(root, i.Interfaces) {
			if isSignificantInterfaceTree(root, i.Interfaces) {
				significantClusters = append(significantClusters, InterfaceCluster{
					Root:       root,
					Depth:      calculateInterfaceDepth(root),
					ClassCount: calculateInterfaceClassCount(root),
				})
			}
		}
	}

	// 显示集群摘要
	i.writeSectionHeader("Significant Interface Hierarchies")
	log.Printf("Interfaces involved in significant hierarchies: %d", len(tree))
	log.Printf("Significant interface clusters: %d", len(significantClusters))
	log.Println()

	// 显示 TopK 最深的集群
	i.showTopKInterfaceClusters(significantClusters, "Deepest", TopK, func(c1, c2 InterfaceCluster) bool {
		return c1.Depth > c2.Depth
	})
	// 显示 TopK 最大的集群
	i.showTopKInterfaceClusters(significantClusters, "Largest Size", TopK, func(c1, c2 InterfaceCluster) bool {
		return c1.ClassCount > c2.ClassCount
	})
	log.Println()

	// 显示集群详情
	if Verbosity > 1 && len(significantClusters) > 0 {
		i.writeSectionHeader("%d Significant Interface Clusters Details:", len(significantClusters))

		for i, cluster := range significantClusters {
			isLast := i == len(significantClusters)-1
			printSignificantInterfaceTree(cluster.Root, 0, isLast, "")
			if !isLast {
				log.Println()
			}
		}
	}
}

func (i *Info) showTopKInterfaceClusters(clusters []InterfaceCluster, label string, k int, less func(InterfaceCluster, InterfaceCluster) bool) {
	sort.Slice(clusters, func(i, j int) bool {
		return less(clusters[i], clusters[j])
	})

	if k > len(clusters) {
		k = len(clusters)
	}

	var cellData [][]string
	for _, cluster := range clusters[:k] {
		cellData = append(cellData, []string{cluster.Root.Name, fmt.Sprintf("%d", cluster.Depth), fmt.Sprintf("%d", cluster.ClassCount)})
	}
	i.writeSectionHeader("Top %d %s Interface Clusters:", k, label)
	i.writeSectionBody(func() {
		tabular.Display([]string{"Root Interface", "Depth", "Cluster Size"}, cellData, false, -1)
	})
}

func isInterface(node *InterfaceNode, interfaces map[string][]string) bool {
	for _, ifaceList := range interfaces {
		for _, iface := range ifaceList {
			if iface == node.Name {
				return true
			}
		}
	}
	return false
}

func printSignificantInterfaceTree(node *InterfaceNode, depth int, isLast bool, prefix string) {
	var nodePrefix string
	if depth > 0 {
		if isLast {
			nodePrefix = prefix + "└── "
		} else {
			nodePrefix = prefix + "├── "
		}
	}

	log.Printf("%s%s\n", nodePrefix, node.Name)

	childPrefix := prefix
	if depth > 0 {
		if isLast {
			childPrefix += "    "
		} else {
			childPrefix += "│   "
		}
	}

	// 显示实现者
	for i, implementor := range node.Implementors {
		isLastImplementor := i == len(node.Implementors)-1 && len(node.Children) == 0
		var implementorPrefix string
		if isLastImplementor {
			implementorPrefix = childPrefix + "└── "
		} else {
			implementorPrefix = childPrefix + "├── "
		}
		log.Printf("%s%s (implementor)\n", implementorPrefix, implementor)
	}

	// 显示子接口
	for i, child := range node.Children {
		isLastChild := i == len(node.Children)-1
		printSignificantInterfaceTree(child, depth+1, isLastChild, childPrefix)
	}
}

func (i *Info) buildInterfaceTree() map[string]*InterfaceNode {
	tree := make(map[string]*InterfaceNode)

	// 创建所有节点
	for class := range i.Inheritance {
		if _, found := tree[class]; !found {
			tree[class] = &InterfaceNode{Name: class}
		}
	}
	for _, interfaces := range i.Interfaces {
		for _, iface := range interfaces {
			if _, found := tree[iface]; !found {
				tree[iface] = &InterfaceNode{Name: iface}
			}
		}
	}

	// 建立继承关系（包括类和接口的继承）
	for class, parents := range i.Inheritance {
		for _, parent := range parents {
			// 确保父节点存在
			if _, found := tree[parent]; !found {
				tree[parent] = &InterfaceNode{Name: parent}
			}
			// 确保子节点存在
			if _, found := tree[class]; !found {
				tree[class] = &InterfaceNode{Name: class}
			}
			// 检查是否已经添加过这个子节点
			alreadyAdded := false
			for _, child := range tree[parent].Children {
				if child.Name == class {
					alreadyAdded = true
					break
				}
			}
			if !alreadyAdded {
				tree[parent].Children = append(tree[parent].Children, tree[class])
			}
		}
	}

	// 建立接口实现关系
	for class, interfaces := range i.Interfaces {
		for _, iface := range interfaces {
			// 确保接口节点存在
			if _, found := tree[iface]; !found {
				tree[iface] = &InterfaceNode{Name: iface}
			}
			// 检查是否已经添加过这个实现者
			alreadyAdded := false
			for _, impl := range tree[iface].Implementors {
				if impl == class {
					alreadyAdded = true
					break
				}
			}
			if !alreadyAdded {
				tree[iface].Implementors = append(tree[iface].Implementors, class)
			}
		}
	}

	return tree
}

func findRootInterfaceClusters(tree map[string]*InterfaceNode) []*InterfaceNode {
	var roots []*InterfaceNode
	for _, node := range tree {
		isRoot := true
		for _, potentialParent := range tree {
			if potentialParent != node {
				for _, child := range potentialParent.Children {
					if child == node {
						isRoot = false
						break
					}
				}
			}
			if !isRoot {
				break
			}
		}
		if isRoot {
			roots = append(roots, node)
		}
	}
	return roots
}

func calculateInterfaceDepth(node *InterfaceNode) int {
	if len(node.Children) == 0 {
		return 1
	}
	maxChildDepth := 0
	for _, child := range node.Children {
		childDepth := calculateInterfaceDepth(child)
		if childDepth > maxChildDepth {
			maxChildDepth = childDepth
		}
	}
	return maxChildDepth + 1
}

func isSignificantInterfaceTree(node *InterfaceNode, interfaces map[string][]string) bool {
	if !isInterface(node, interfaces) {
		return false
	}
	if len(node.Children) > significantInheritanceChildren {
		return true
	}
	if calculateInterfaceDepth(node) > significantInheritanceDepth {
		return true
	}
	if len(node.Implementors) > significantInheritanceChildren {
		return true
	}
	for _, child := range node.Children {
		if isSignificantInterfaceTree(child, interfaces) {
			return true
		}
	}
	return false
}

func calculateInterfaceClassCount(node *InterfaceNode) int {
	count := len(node.Implementors)
	for _, child := range node.Children {
		count += calculateInterfaceClassCount(child)
	}
	return count
}
