package ast

import (
	"fmt"
	"sort"

	"github.com/emirpasic/gods/trees/redblacktree"
	"github.com/fatih/color"
)

type ClassNode struct {
	Name     string
	Children []*ClassNode
}

func (i *Info) showInheritanceReport() {
	color.Magenta("Significant Class Inheritance Hierarchies:")

	tree := buildInheritanceTree(i.Inheritance)
	rootNodes := findRootNodes(tree)

	significantHierarchies := []*ClassNode{}
	maxDepth := 0

	for _, root := range rootNodes {
		depth := calculateDepth(root)
		if depth > maxDepth {
			maxDepth = depth
		}
		if isSignificantTree(root) {
			significantHierarchies = append(significantHierarchies, root)
		}
	}

	// Render summary
	color.Magenta("  - Total classes: %d", len(i.Classes))
	color.Magenta("  - Classes involved in inheritance: %d", countClassesWithInheritance(tree))
	color.Magenta("  - Maximum inheritance depth: %d\n", maxDepth)
	fmt.Println()

	// Render the tree
	if len(significantHierarchies) > 0 {
		for i, root := range significantHierarchies {
			isLast := i == len(significantHierarchies)-1
			printSignificantTree(root, 0, isLast, "")
			if !isLast {
				fmt.Println() // Add a blank line between major inheritance trees
			}
		}
	} else {
		fmt.Println("  No inheritance hierarchies with more than 2 children or depth > 2 found.")
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

func findRootNodes(tree *redblacktree.Tree) []*ClassNode {
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

	fmt.Printf("%s%s\n", nodePrefix, node.Name)

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
