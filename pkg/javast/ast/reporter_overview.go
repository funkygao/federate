package ast

func (i *Info) showOverviewReport() (empty bool) {
	if Verbosity > 2 {
		i.showFileStatsReport(TopK)
		i.showNameCountSection("Annotations", []string{"Annotation"}, topN(i.Annotations, TopK))
		i.showNameCountSection("Imports", []string{"Import"}, topN(i.Imports, TopK))
	}

	i.showNameCountSection("Methods", []string{"NonStatic Declaration", "Static Declaration", "Call"},
		topN(i.Methods, TopK), topN(i.StaticMethodDeclarations, TopK), topN(i.MethodCalls, TopK))
	log.Println()
	i.showNameCountSection("Variables", []string{"Declaration", "Reference"}, topN(i.Variables, TopK), topN(i.VariableReferences, TopK))

	log.Printf("\nTotal classes: %d, methods: %d, variables: %d, variable references: %d, method calls: %d",
		len(i.Classes), len(i.Methods), len(i.Variables), len(i.VariableReferences), len(i.MethodCalls))

	return
}
