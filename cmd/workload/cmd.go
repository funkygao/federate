package workload

import (
	"github.com/spf13/cobra"
)

var CmdGroup = &cobra.Command{
	Use:   "workload",
	Short: "Workload safeguarding and governance",
	Long: `Workload safeguarding and governance mechanisms for system resource protection.

Ensures system stability, performance, and reliability under diverse workload conditions through:
- Overload protection
- Self-healing
- Large payload protection

  Find more information at: https://joyspace.jd.com/pages/yT1mL418bTf8itXDoRFL
`,
}

func init() {
	CmdGroup.AddCommand(integrationCmd, heterogeneousCmd)
}
