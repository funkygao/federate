package ast

import (
	"fmt"
)

func (i *Info) showLambdaReport() {
	// 行数分布
	lineCounts := make(map[string]int)
	for _, info := range i.LambdaInfos {
		if info.LineCount == 1 {
			lineCounts["Single-line"]++
		} else {
			lineCounts["Multi-line"]++
		}
	}
	i.showNameCountSection("Lambda Complexity", []string{"Complexity"}, topN(mapToNameCount(lineCounts), TopK))

	// 参数数量分布
	paramCounts := make(map[string]int)
	for _, info := range i.LambdaInfos {
		paramCounts[fmt.Sprintf("%d params", info.ParameterCount)]++
	}
	i.showNameCountSection("Lambda Parameters", []string{"Parameter Count"}, topN(mapToNameCount(paramCounts), TopK))

	// 最常见的上下文
	contexts := make(map[string]int)
	for _, info := range i.LambdaInfos {
		contexts[info.Context]++
	}
	i.showNameCountSection("Top Lambda Contexts", []string{"Context"}, topN(mapToNameCount(contexts), TopK))

	// 与 Stream 操作的关联
	streamOps := make(map[string]int)
	for _, info := range i.LambdaInfos {
		streamOps[info.AssociatedStreamOp]++
	}
	i.showNameCountSection("Top Associated Stream Operations", []string{"Operation"}, topN(mapToNameCount(streamOps), TopK))

	// 模式分布
	patterns := make(map[string]int)
	for _, info := range i.LambdaInfos {
		patterns[info.Pattern]++
	}
	i.showNameCountSection("Lambda Patterns", []string{"Pattern"}, topN(mapToNameCount(patterns), TopK))
}
