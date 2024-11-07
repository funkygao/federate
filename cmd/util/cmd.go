package util

import (
	"federate/cmd/github"
	"github.com/spf13/cobra"
)

var CmdGroup = &cobra.Command{
	Use:   "util",
	Short: "Utilities for developers",
	Long:  `The util command group provides a set of commands for developers.`,
}

func init() {
	CmdGroup.AddCommand(ygrepCmd, tlaplusCmd, github.CmdGroup, tfidfCmd, inventoryCmd, allCmd)
}
