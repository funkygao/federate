package insight

import (
	"log"

	"federate/pkg/ast/listener"
	"federate/pkg/util"
	"github.com/spf13/cobra"
)

var methodsCmd = &cobra.Command{
	Use:   "methods <dir or file>",
	Short: "Count the total number of methods in Java source files",
	Long:  `This command recursively analyzes Java source files, counting the total number of methods while excluding test files.`,
	Args:  cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		showMethodCount(args[0])
	},
}

func showMethodCount(path string) {
	l := listener.NewMethodCountListner()
	if util.IsDir(path) {
		if err := parseDir(path, l); err != nil {
			log.Fatalf("Error parsing directory: %v", err)
		}
	} else if err := parseFile(path, l); err != nil {
		log.Fatalf("Error parsing directory: %v", err)
	}

	log.Printf("Total number of methods: %d", l.MethodCount)
}

func init() {
	methodsCmd.Flags().BoolVarP(&debug, "debug", "d", true, "Debug mode")
	methodsCmd.Flags().BoolVarP(&pprof, "pprof", "p", false, "Enable pprof")
}
