package mybatis

import (
	"fmt"
	"sort"
	"strconv"

	"federate/pkg/tabular"
)

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func sortMapByValue(m map[string]int) [][]string {
	type kv struct {
		Key   string
		Value int
	}

	var ss []kv
	for k, v := range m {
		ss = append(ss, kv{k, v})
	}

	sort.Slice(ss, func(i, j int) bool {
		return ss[i].Value > ss[j].Value
	})

	var result [][]string
	for _, kv := range ss {
		result = append(result, []string{kv.Key, fmt.Sprintf("%d", kv.Value)})
	}

	return result
}

func sortNestedMapByValueDesc(m map[string]map[string]int) [][]string {
	type innerKV struct {
		Key   string
		Value int
	}

	type outerKV struct {
		Key   string
		Inner []innerKV
	}

	var outerSlice []outerKV
	for outerKey, innerMap := range m {
		var innerSlice []innerKV
		for innerKey, value := range innerMap {
			innerSlice = append(innerSlice, innerKV{innerKey, value})
		}

		// 对内层 slice 按 Value 降序排序，Value 相同时按 Key 字母顺序排序
		sort.Slice(innerSlice, func(i, j int) bool {
			if innerSlice[i].Value == innerSlice[j].Value {
				return innerSlice[i].Key < innerSlice[j].Key
			}
			return innerSlice[i].Value > innerSlice[j].Value
		})

		outerSlice = append(outerSlice, outerKV{outerKey, innerSlice})
	}

	// 对外层 slice 按字母顺序排序
	sort.Slice(outerSlice, func(i, j int) bool {
		return outerSlice[i].Key < outerSlice[j].Key
	})

	// 构建结果
	var result [][]string
	for _, outer := range outerSlice {
		for i, inner := range outer.Inner {
			var row []string
			if i == 0 {
				row = []string{outer.Key, inner.Key, strconv.Itoa(inner.Value)}
			} else {
				row = []string{"", inner.Key, strconv.Itoa(inner.Value)}
			}
			result = append(result, row)
		}
	}

	return result
}

func printTopN(m map[string]int, topK int) {
	if len(m) < topK {
		topK = len(m)
	}

	items := make([]tabular.BarChartItem, 0, len(m))
	for key, count := range m {
		items = append(items, tabular.BarChartItem{Name: key, Count: count})
	}
	tabular.PrintBarChart(items, topK)
}
