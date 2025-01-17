package ast

func (i *Info) PrepareData() {
	i.prepareSignificantInheritanceClusters()
	i.prepareSignificantInterfaceClusters()
	i.prepareRelations()
}

func (i *Info) prepareSignificantInheritanceClusters() {
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

	// Store the significant clusters in Info struct
	i.SignificantInheritanceClusters = significantClusters
}

func (i *Info) prepareSignificantInterfaceClusters() {
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

	// Store the significant clusters in Info struct
	i.SignificantInterfaceClusters = significantClusters
}

func (i *Info) prepareRelations() {
	var relationships []ClusterRelationship

	// Build maps of cluster roots for quick lookup
	inheritanceRoots := make(map[string]*ClassNode)
	for _, cluster := range i.SignificantInheritanceClusters {
		inheritanceRoots[cluster.Root.Name] = cluster.Root
	}

	interfaceRoots := make(map[string]*InterfaceNode)
	for _, cluster := range i.SignificantInterfaceClusters {
		interfaceRoots[cluster.Root.Name] = cluster.Root
	}

	// Check for inheritance cluster roots implementing interface cluster roots
	for _, inhCluster := range i.SignificantInheritanceClusters {
		rootClass := inhCluster.Root.Name

		// Check if rootClass implements any significant interface cluster root
		if interfaces, ok := i.Interfaces[rootClass]; ok {
			for _, iface := range interfaces {
				if _, isSignificant := interfaceRoots[iface]; isSignificant {
					relationships = append(relationships, ClusterRelationship{
						FromCluster: rootClass,
						Relation:    "implements",
						ToCluster:   iface,
					})
				}
			}
		}
	}

	// Check for inheritance among cluster roots
	for _, inhCluster := range i.SignificantInheritanceClusters {
		rootClass := inhCluster.Root.Name

		if superclasses, ok := i.Inheritance[rootClass]; ok {
			for _, superclass := range superclasses {
				if _, isSignificant := inheritanceRoots[superclass]; isSignificant {
					relationships = append(relationships, ClusterRelationship{
						FromCluster: rootClass,
						Relation:    "extends",
						ToCluster:   superclass,
					})
				}
			}
		}
	}

	// Check if interface cluster roots extend other interface cluster roots
	for _, intfCluster := range i.SignificantInterfaceClusters {
		rootInterface := intfCluster.Root.Name

		if superinterfaces, ok := i.Inheritance[rootInterface]; ok {
			for _, superinterface := range superinterfaces {
				if _, isSignificant := interfaceRoots[superinterface]; isSignificant {
					relationships = append(relationships, ClusterRelationship{
						FromCluster: rootInterface,
						Relation:    "extends",
						ToCluster:   superinterface,
					})
				}
			}
		}
	}

	// Check for compositions where either the containing class or the composed class is a significant cluster root
	for _, comp := range i.Compositions {
		containingClass := comp.ContainingClass
		composedClass := comp.ComposedClass

		// Check if both the containing and composed classes are significant cluster roots
		isContainingClassSignificant := inheritanceRoots[containingClass] != nil || interfaceRoots[containingClass] != nil
		isComposedClassSignificant := inheritanceRoots[composedClass] != nil || interfaceRoots[composedClass] != nil

		if isContainingClassSignificant && isComposedClassSignificant {
			relationships = append(relationships, ClusterRelationship{
				FromCluster: containingClass,
				Relation:    "composes",
				ToCluster:   composedClass,
			})
		}
	}

	i.Relations = relationships
}
