package ast

import (
	"fmt"
	"log"
	"sort"

	"federate/pkg/tabular"
	"github.com/emirpasic/gods/trees/redblacktree"
)

type ClassNode struct {
	Name     string
	Children []*ClassNode
}

type InheritanceCluster struct {
	Root       *ClassNode
	Depth      int
	ClassCount int
}

func (i *Info) showInheritanceReport() {
	tree := buildInheritanceTree(i.Inheritance)

	significantClusters := []InheritanceCluster{}
	for _, root := range findRootClusters(tree) {
		if isSignificantTree(root) {
			significantClusters = append(significantClusters, InheritanceCluster{
				Root:       root,
				Depth:      calculateDepth(root),
				ClassCount: calculateClassCount(root),
			})
		}
	}

	// Render summary
	i.writeSectionHeader("Significant Class Inheritance Hierarchies")
	log.Printf("Classes involved in inheritance: %d", countClassesWithInheritance(tree))
	log.Printf("Significant inheritance clusters: %d", len(significantClusters))
	log.Println()

	// Show TopK deepest clusters
	i.showTopKClusters(significantClusters, "Deepest", TopK, func(c1, c2 InheritanceCluster) bool {
		return c1.Depth > c2.Depth
	})
	// Show TopK largest clusters
	i.showTopKClusters(significantClusters, "Largest Size", TopK, func(c1, c2 InheritanceCluster) bool {
		return c1.ClassCount > c2.ClassCount
	})
	log.Println()

	// Render the clusters
	if Verbosity > 1 && len(significantClusters) > 0 {
		i.writeSectionHeader("%d Significant Class Inheritance Clusters Details:", len(significantClusters))

		for i, cluster := range significantClusters {
			isLast := i == len(significantClusters)-1
			printSignificantTree(cluster.Root, 0, isLast, "")
			if !isLast {
				log.Println()
			}
		}
	}
}

func buildInheritanceTree(inheritance map[string][]string) *redblacktree.Tree {
	tree := redblacktree.NewWithStringComparator()

	// First pass: create all nodes
	for class := range inheritance {
		if _, found := tree.Get(class); !found {
			tree.Put(class, &ClassNode{Name: class})
		}
	}

	// Second pass: build relationships
	for class, superclasses := range inheritance {
		node, _ := tree.Get(class)
		classNode := node.(*ClassNode)

		for _, superclass := range superclasses {
			superNode, found := tree.Get(superclass)
			if !found {
				superNode = &ClassNode{Name: superclass}
				tree.Put(superclass, superNode)
			}
			superClassNode := superNode.(*ClassNode)
			superClassNode.Children = append(superClassNode.Children, classNode)
		}
	}

	return tree
}

func findRootClusters(tree *redblacktree.Tree) []*ClassNode {
	roots := []*ClassNode{}
	it := tree.Iterator()
	for it.Next() {
		node := it.Value().(*ClassNode)
		isRoot := true
		it2 := tree.Iterator()
		for it2.Next() {
			parent := it2.Value().(*ClassNode)
			for _, child := range parent.Children {
				if child.Name == node.Name {
					isRoot = false
					break
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

func calculateDepth(node *ClassNode) int {
	if len(node.Children) == 0 {
		return 1
	}
	maxChildDepth := 0
	for _, child := range node.Children {
		childDepth := calculateDepth(child)
		if childDepth > maxChildDepth {
			maxChildDepth = childDepth
		}
	}
	return maxChildDepth + 1
}

func isSignificantTree(node *ClassNode) bool {
	if len(node.Children) > significantInheritanceChildren {
		return true
	}
	if calculateDepth(node) > significantInheritanceDepth {
		return true
	}
	for _, child := range node.Children {
		if isSignificantTree(child) {
			return true
		}
	}
	return false
}

func printSignificantTree(node *ClassNode, depth int, isLast bool, prefix string) {
	var nodePrefix string
	if depth > 0 {
		if isLast {
			nodePrefix = prefix + "└── "
		} else {
			nodePrefix = prefix + "├── "
		}
	}

	log.Printf("%s%s\n", nodePrefix, node.Name)

	sort.Slice(node.Children, func(i, j int) bool {
		return node.Children[i].Name < node.Children[j].Name
	})

	childPrefix := prefix
	if depth > 0 {
		if isLast {
			childPrefix += "    "
		} else {
			childPrefix += "│   "
		}
	}

	for i, child := range node.Children {
		isLastChild := i == len(node.Children)-1
		printSignificantTree(child, depth+1, isLastChild, childPrefix)
	}
}

func countClassesWithInheritance(tree *redblacktree.Tree) int {
	count := 0
	it := tree.Iterator()
	for it.Next() {
		node := it.Value().(*ClassNode)
		key := it.Key().(string)
		if len(node.Children) > 0 || key != node.Name {
			count++
		}
	}
	return count
}

func (i *Info) showTopKClusters(clusters []InheritanceCluster, label string, k int, less func(InheritanceCluster, InheritanceCluster) bool) {
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
	i.writeSectionHeader("Top %d %s Inheritance Clusters:", k, label)
	tabular.Display([]string{"Root Class", "Depth", "Class Count"}, cellData, false, -1)
}

func calculateClassCount(node *ClassNode) int {
	count := 1 // 计入当前节点
	for _, child := range node.Children {
		count += calculateClassCount(child)
	}
	return count
}
