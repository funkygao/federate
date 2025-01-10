package insight

import (
	"github.com/spf13/cobra"
)

var CmdGroup = &cobra.Command{
	Use:   "insight",
	Short: "Extract valuable insights from source code",
}

func init() {
	CmdGroup.AddCommand(taxonomyCmd, extensionCmd, mybatisCmd, gitInsightCmd, tfidfCmd)
}
