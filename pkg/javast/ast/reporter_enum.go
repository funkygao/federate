package ast

import (
	"fmt"
	"sort"
	"strings"

	"federate/pkg/tabular"
	"federate/pkg/util"
)

func (i *Info) showEnumReport() (empty bool) {
	if len(i.Enums) == 0 {
		return true
	}

	maxWidth := 0
	for _, enum := range i.Enums {
		maxWidth = max(maxWidth, util.TerminalDisplayWidth(enum.Name))
	}

	var cellData [][]string
	for _, enum := range i.Enums {
		sort.Strings(enum.Values)
		cellData = append(cellData, []string{
			enum.Name,
			fmt.Sprintf("%d", len(enum.Values)),
			util.Truncate(strings.Join(enum.Values, "/"), util.TerminalWidth()-maxWidth-15),
		})
	}

	// 对 cellData 按 Enum Name 排序
	sort.Slice(cellData, func(i, j int) bool {
		return cellData[i][0] < cellData[j][0]
	})

	i.writeSectionHeader("%d Enum Analysis", len(i.Enums))
	i.writeSectionBody(func() {
		tabular.Display([]string{"Enum Name", "N", "Values"}, cellData, false, -1)
	})

	return
}
