package ast

import (
	"sort"

	"federate/pkg/tabular"
)

func (i *Info) showClusterRelationships() (empty bool) {
	// Sort i.Relations for better presentation
	sort.Slice(i.Relations, func(a, b int) bool {
		if i.Relations[a].FromCluster != i.Relations[b].FromCluster {
			return i.Relations[a].FromCluster < i.Relations[b].FromCluster
		}
		if i.Relations[a].Relation != i.Relations[b].Relation {
			return i.Relations[a].Relation < i.Relations[b].Relation
		}
		return i.Relations[a].ToCluster < i.Relations[b].ToCluster
	})

	// Prepare data for display
	var cellData [][]string
	for _, rel := range i.Relations {
		cellData = append(cellData, []string{
			rel.FromCluster,
			rel.Relation,
			rel.ToCluster,
		})
	}

	if len(cellData) > 0 {
		i.writeSectionHeader("Relations Between Significant Clusters")
		i.writeSectionBody(func() {
			tabular.Display([]string{"Inheritance Cluster Root", "Relation", "Interface Cluster Root"}, cellData, false, -1)
		})
	}

	return
}
