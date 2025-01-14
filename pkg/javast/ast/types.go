package ast

import (
	"sort"

	"federate/pkg/primitive"
	"federate/pkg/prompt"
)

type LambdaInfo struct {
	LineCount          int    `json:"lineCount"`
	ParameterCount     int    `json:"parameterCount"`
	Context            string `json:"context"`
	AssociatedStreamOp string `json:"associatedStreamOp"`
	Pattern            string `json:"pattern"`
}

type FileStats struct {
	FileName       string `json:"fileName"`
	NetLinesOfCode int    `json:"netLinesOfCode"`
	MethodCount    int    `json:"methodCount"`
	FieldCount     int    `json:"fieldCount"`
}

type CompositionInfo struct {
	ContainingClass string `json:"containingClass"`
	ComposedClass   string `json:"composedClass"`
	FieldName       string `json:"fieldName"`
}

type ComplexCondition struct {
	FileName   string `json:"fileName"`
	MethodName string `json:"methodName"`
	Condition  string `json:"condition"`
	Complexity int    `json:"complexity"`
	LineNumber int    `json:"lineNumber"`
}

type ComplexLoop struct {
	MethodName   string `json:"methodName"`
	FileName     string `json:"fileName"`
	LoopType     string `json:"loopType"`
	LineNumber   int    `json:"lineNumber"`
	NestingLevel int    `json:"nestingLevel"`
	BodySize     int    `json:"bodySize"`
}

type FunctionalUsage struct {
	MethodName string `json:"methodName"`
	FileName   string `json:"fileName"`
	LineNumber int    `json:"lineNumber"`
	Type       string `json:"type"`
	Operation  string `json:"operation"`
	Context    string `json:"context"`
}

type Info struct {
	logger *prompt.PromptLogger `json:"-"`

	Imports            []string             `json:"imports"`
	Classes            []string             `json:"classes"`
	Methods            []string             `json:"methods"`
	Variables          []string             `json:"variables"`
	VariableReferences []string             `json:"variableReferences"`
	MethodCalls        []string             `json:"methodCalls"`
	Inheritance        map[string][]string  `json:"inheritance"`
	Interfaces         map[string][]string  `json:"interfaces"`
	Annotations        []string             `json:"annotations"`
	ComplexConditions  []ComplexCondition   `json:"complexConditions"`
	Compositions       []CompositionInfo    `json:"compositions"`
	ComplexLoops       []ComplexLoop        `json:"complexLoops"`
	FunctionalUsages   []FunctionalUsage    `json:"functionalUsages"`
	LambdaInfos        []LambdaInfo         `json:"lambdaInfos"`
	FileStats          map[string]FileStats `json:"fileStats"`
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

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
