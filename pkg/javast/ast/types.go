package ast

import (
	"sort"

	"federate/pkg/primitive"
)

type Info struct {
	Imports     []string            `json:"imports"`
	Classes     []string            `json:"classes"`
	Methods     []string            `json:"methods"`
	Variables   []string            `json:"variables"`
	MethodCalls []string            `json:"methodCalls"`
	Inheritance map[string][]string `json:"inheritance"`
	Interfaces  map[string][]string `json:"interfaces"`
	Annotations []string            `json:"annotations"`
}

func topN(items any, n int) []primitive.NameCount {
	var result []primitive.NameCount

	switch v := items.(type) {
	case []string:
		counts := make(map[string]int)
		for _, item := range v {
			counts[item]++
		}
		for name, count := range counts {
			result = append(result, primitive.NameCount{Name: name, Count: count})
		}
	case []primitive.NameCount:
		result = v
	}

	sort.Slice(result, func(i, j int) bool {
		return result[i].Count > result[j].Count
	})

	if n > len(result) {
		n = len(result)
	}
	return result[:n]
}

func mapToNameCount(m map[string]int) []primitive.NameCount {
	var result []primitive.NameCount
	for k, v := range m {
		result = append(result, primitive.NameCount{Name: k, Count: v})
	}
	return result
}
