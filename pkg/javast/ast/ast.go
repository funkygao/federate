package ast

import "federate/pkg/prompt"

type Info struct {
	logger *prompt.PromptLogger `json:"-"`

	Imports                  []string             `json:"imports"`
	Classes                  []string             `json:"classes"`
	Methods                  []string             `json:"methods"` // 非静态方法声明
	StaticMethodDeclarations []string             `json:"staticMethodDeclarations"`
	MethodCalls              []string             `json:"methodCalls"`
	Variables                []string             `json:"variables"`
	VariableReferences       []string             `json:"variableReferences"`
	Inheritance              map[string][]string  `json:"inheritance"`
	Interfaces               map[string][]string  `json:"interfaces"`
	Annotations              []string             `json:"annotations"`
	ComplexConditions        []ComplexCondition   `json:"complexConditions"`
	Compositions             []CompositionInfo    `json:"compositions"`
	ComplexLoops             []ComplexLoop        `json:"complexLoops"`
	FunctionalUsages         []FunctionalUsage    `json:"functionalUsages"`
	LambdaInfos              []LambdaInfo         `json:"lambdaInfos"`
	FileStats                map[string]FileStats `json:"fileStats"`
	ReflectionUsages         []ReflectionUsage    `json:"reflectionUsages"`
	TransactionInfos         []TransactionInfo    `json:"transactionInfos"`
	ExceptionCatches         []ExceptionCatchInfo `json:"exceptionCatches"`
	MethodThrows             []MethodThrowsInfo   `json:"methodThrows"`
	ConcurrencyUsages        []ConcurrencyUsage   `json:"concurrencyUsages"`
	Enums                    []EnumInfo           `json:"enums"`

	SignificantInheritanceClusters []InheritanceCluster `json:"-"`
	SignificantInterfaceClusters   []InterfaceCluster   `json:"-"`
	Relations                      []ClusterRelationship
}
