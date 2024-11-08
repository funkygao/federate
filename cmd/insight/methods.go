package insight

import (
	"log"

	"federate/pkg/ast/listener"
	"github.com/spf13/cobra"
)

var methodsCmd = &cobra.Command{
	Use:   "methods <dir>",
	Short: "Count the total number of methods in Java source files",
	Long:  `This command recursively analyzes Java source files in the specified directory, counting the total number of methods while excluding test files.`,
	Args:  cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		showMethodCount(args[0])
	},
}

func showMethodCount(dir string) {
	l := listener.NewMethodCountListner()
	if err := parseDir(dir, l); err != nil {
		log.Fatalf("Error parsing directory: %v", err)
	}

	log.Printf("Total number of methods: %d", l.MethodCount)
}
