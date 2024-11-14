package util

import (
	"log"

	"federate/pkg/javast"
	"github.com/spf13/cobra"
)

var javastCmd = &cobra.Command{
	Use:   "javast <dir>",
	Short: "Parse Java AST",
	Args:  cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		parseJava(args[0])
	},
}

func parseJava(rootDir string) {
	r, err := javast.RecursiveParse("parse", rootDir)
	if err != nil {
		log.Fatalf("%v", err)
	}

	log.Printf("%+v", r)
}
