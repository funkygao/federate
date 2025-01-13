package ast

import (
	"sort"

	"federate/pkg/primitive"
)

type Info struct {
	Imports     []string `json:"imports"`
	Classes     []string `json:"classes"`
	Methods     []string `json:"methods"`
	Variables   []string `json:"variables"`
	MethodCalls []string `json:"methodCalls"`
}

func topN(items []string, n int) []primitive.NameCount {
	counts := make(map[string]int)
	for _, item := range items {
		counts[item]++
	}

	var result []primitive.NameCount
	for name, count := range counts {
		result = append(result, primitive.NameCount{Name: name, Count: count})
	}

	sort.Slice(result, func(i, j int) bool {
		return result[i].Count > result[j].Count
	})

	if n > len(result) {
		n = len(result)
	}
	return result[:n]
}
