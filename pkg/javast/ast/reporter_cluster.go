package ast

import (
	"sort"

	"federate/pkg/tabular"
)

func (i *Info) showClusterRelationships() {
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

	// Sort relationships for better presentation
	sort.Slice(relationships, func(a, b int) bool {
		if relationships[a].FromCluster != relationships[b].FromCluster {
			return relationships[a].FromCluster < relationships[b].FromCluster
		}
		if relationships[a].Relation != relationships[b].Relation {
			return relationships[a].Relation < relationships[b].Relation
		}
		return relationships[a].ToCluster < relationships[b].ToCluster
	})

	// Prepare data for display
	var cellData [][]string
	for _, rel := range relationships {
		cellData = append(cellData, []string{
			rel.FromCluster,
			rel.Relation,
			rel.ToCluster,
		})
	}

	if len(cellData) > 0 {
		i.writeSectionHeader("Relationships Between Significant Clusters")
		i.writeSectionBody(func() {
			tabular.Display([]string{"Inheritance Cluster Root", "Relation", "Interface Cluster Root"}, cellData, false, -1)
		})
	}
}
