package ast

import (
	"embed"
	"encoding/json"
	"fmt"
	"log"
	"net/http"

	"federate/pkg/util"
)

//go:embed static
var static embed.FS

func (i *Info) StartWebServer(port string) {
	http.HandleFunc("/", handleRoot)
	http.HandleFunc("/report", i.handleReport)

	log.Printf("Starting web server on http://localhost:%s", port)
	go func() {
		if err := http.ListenAndServe(":"+port, nil); err != nil {
			log.Fatalf("Failed to start server: %v", err)
		}
	}()

	util.OpenBrowser(fmt.Sprintf("http://localhost:%s", port))

	// Wait for user to press Ctrl+C
	select {}
}

func handleRoot(w http.ResponseWriter, r *http.Request) {
	content, err := static.ReadFile("static/index.html")
	if err != nil {
		http.Error(w, "Could not read index.html", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/html")
	w.Write(content)
}

func (i *Info) handleReport(w http.ResponseWriter, r *http.Request) {
	report := i.GenerateReportJSON()

	jsonData, err := json.Marshal(report)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.Write(jsonData)
}

func (i *Info) GenerateReportJSON() map[string]interface{} {
	return map[string]interface{}{
		"fileStats":   i.FileStats,
		"annotations": topN(i.Annotations, TopK),
		"imports":     topN(i.Imports, TopK),

		"classes": topN(i.Classes, TopK),
		"methods": map[string]interface{}{
			"nonStatic": topN(i.Methods, TopK),
			"static":    topN(i.StaticMethodDeclarations, TopK),
			"calls":     topN(i.MethodCalls, TopK),
		},
		"variables": map[string]interface{}{
			"declarations": topN(i.Variables, TopK),
			"references":   topN(i.VariableReferences, TopK),
		},

		"inheritance": map[string]interface{}{
			"relationships": i.Inheritance,
			"clusters":      i.SignificantInheritanceClusters,
		},
		"interfaces": map[string]interface{}{
			"implementations": i.Interfaces,
			"clusters":        i.SignificantInterfaceClusters,
		},
		"relations":    i.Relations,
		"compositions": i.Compositions,

		"complexConditions": i.ComplexConditions,
		"complexLoops":      i.ComplexLoops,

		"reflectionUsages": i.ReflectionUsages,
		"functionalUsages": i.FunctionalUsages,
		"lambdaInfos":      i.LambdaInfos,

		"methodThrows":      i.MethodThrows,
		"exceptionCatches":  i.ExceptionCatches,
		"transactions":      i.TransactionInfos,
		"concurrencyUsages": i.ConcurrencyUsages,
	}
}
