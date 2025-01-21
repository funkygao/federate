package insight

import (
	"github.com/spf13/cobra"
)

var CmdGroup = &cobra.Command{
	Use:   "insight",
	Short: "Extract valuable insights from source code",
}

func init() {
	CmdGroup.AddCommand(apiCmd, astCmd, mybatisCmd, refCmd, gitInsightCmd, taxonomyCmd, extensionCmd, architectureCmd, tfidfCmd)
}
