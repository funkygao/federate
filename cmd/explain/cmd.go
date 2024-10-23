package explain

import (
	"github.com/spf13/cobra"
)

var CmdGroup = &cobra.Command{
	Use:   "explain",
	Short: "Describes microservice fusion key mechanisms",
	Long:  `The explain command describes microservice fusion key mechanisms`,
}

func init() {
	CmdGroup.AddCommand(conventionCmd, taintCmd, assumptionCmd)
}
