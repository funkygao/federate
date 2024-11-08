package insight

import (
	"log"

	"federate/pkg/ast/parser"
	"github.com/spf13/cobra"
)

var methodsCmd = &cobra.Command{
	Use:   "methods <dir>",
	Short: "Count the total number of methods in Java source files",
	Long:  `This command analyzes Java source files in the specified directory and its subdirectories, counting the total number of methods while excluding test files.`,
	Args:  cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		showMethodCount(args[0])
	},
}

func showMethodCount(dir string) {
	listener := &methodCountListener{}
	err := parseDir(dir, listener)
	if err != nil {
		log.Fatalf("Error parsing directory: %v", err)
	}

	log.Printf("Total number of methods: %d", listener.methodCount)
}

type methodCountListener struct {
	parser.BaseJava8ParserListener
	methodCount int
}

func (l *methodCountListener) EnterMethodDeclaration(ctx *parser.MethodDeclarationContext) {
	l.methodCount++
}
