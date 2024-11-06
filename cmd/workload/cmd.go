package workload

import (
	"github.com/spf13/cobra"
)

var CmdGroup = &cobra.Command{
	Use:   "workload",
	Short: "Workload governance and protection",
	Long: `Workload governance and protection mechanisms for system resource safeguarding.
Ensures system stability and performance under diverse workload conditions.

Features include:
- Overload protection
- Self-healing
- Large payload protection

  Find more information at: https://joyspace.jd.com/pages/yT1mL418bTf8itXDoRFL
`,
}

func init() {
	CmdGroup.AddCommand(integrationCmd)
}
