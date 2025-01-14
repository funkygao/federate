package ast

import (
	"fmt"
	"path/filepath"
	"sort"
	"strings"

	"federate/pkg/tabular"
)

func (i *Info) showFileStatsReport(topN int) {
	i.writeSectionHeader("Top %d Files by Different Metrics:", topN)

	var fileStatsList []FileStats
	for _, stats := range i.FileStats {
		stats.FileName = strings.TrimSuffix(filepath.Base(stats.FileName), ".java")
		fileStatsList = append(fileStatsList, stats)
	}

	// 按字段数排序
	sort.Slice(fileStatsList, func(i, j int) bool {
		return fileStatsList[i].FieldCount > fileStatsList[j].FieldCount
	})
	fieldSortedData := getTopNDataByFields(fileStatsList, topN)

	// 按方法数排序
	sort.Slice(fileStatsList, func(i, j int) bool {
		return fileStatsList[i].MethodCount > fileStatsList[j].MethodCount
	})
	methodSortedData := getTopNDataByMethods(fileStatsList, topN)

	// 按净代码行数排序
	sort.Slice(fileStatsList, func(i, j int) bool {
		return fileStatsList[i].NetLinesOfCode > fileStatsList[j].NetLinesOfCode
	})
	locSortedData := getTopNDataByLOC(fileStatsList, topN)

	// 准备表格数据
	var combinedData [][]string
	headers := []string{
		"Top by Fields", "Fields",
		"Top by Methods", "Methods",
		"Top by LOC", "LOC",
	}

	for i := 0; i < topN; i++ {
		row := make([]string, 0, 6)
		if i < len(fieldSortedData) {
			row = append(row, fieldSortedData[i]...)
		} else {
			row = append(row, "", "")
		}
		if i < len(methodSortedData) {
			row = append(row, methodSortedData[i]...)
		} else {
			row = append(row, "", "")
		}
		if i < len(locSortedData) {
			row = append(row, locSortedData[i]...)
		} else {
			row = append(row, "", "")
		}
		combinedData = append(combinedData, row)
	}

	tabular.Display(headers, combinedData, false, -1)
}

func getTopNDataByFields(stats []FileStats, n int) [][]string {
	var data [][]string
	for i := 0; i < n && i < len(stats); i++ {
		data = append(data, []string{
			stats[i].FileName,
			fmt.Sprintf("%d", stats[i].FieldCount),
		})
	}
	return data
}

func getTopNDataByMethods(stats []FileStats, n int) [][]string {
	var data [][]string
	for i := 0; i < n && i < len(stats); i++ {
		data = append(data, []string{
			stats[i].FileName,
			fmt.Sprintf("%d", stats[i].MethodCount),
		})
	}
	return data
}

func getTopNDataByLOC(stats []FileStats, n int) [][]string {
	var data [][]string
	for i := 0; i < n && i < len(stats); i++ {
		data = append(data, []string{
			stats[i].FileName,
			fmt.Sprintf("%d", stats[i].NetLinesOfCode),
		})
	}
	return data
}
