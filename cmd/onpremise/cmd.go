package onpremise

import (
	"github.com/spf13/cobra"
)

var (
	CmdGroup = &cobra.Command{
		Use:   "on-premise",
		Short: "On-premise deployment and management toolkit",
		Long:  "A set of tools for managing and maintaining on-premise deployments of microservices.",
	}

	serverCmd = &cobra.Command{
		Use:   "server",
		Short: "Start the on-premise management server",
		Long:  "Start the on-premise management server for local operations and maintenance of microservices.",
		Run: func(cmd *cobra.Command, args []string) {
		},
	}

	dbPurgeCmd = &cobra.Command{
		Use:   "purge",
		Short: "Purge expired database records",
		Long:  "Periodically clean up and remove expired records from the database to maintain optimal performance.",
		Run: func(cmd *cobra.Command, args []string) {
			doPurgeDb()
		},
	}

	totpCmd = &cobra.Command{
		Use:   "totp",
		Short: "Generate TOTP for node shutdown",
		Long:  "Generate a Time-based One-Time Password (TOTP) for authorizing the shutdown of microservice nodes.",
		Run: func(cmd *cobra.Command, args []string) {
			generateTOTP()
		},
	}
)

func init() {
	CmdGroup.AddCommand(serverCmd, dbPurgeCmd, totpCmd)

	dbPurgeCmd.Flags().StringVarP(&mysqlHost, "mysql-host", "H", "master.mysql.local", "MySQL Host")
	dbPurgeCmd.Flags().StringVarP(&mysqlPort, "mysql-port", "p", "3358", "MySQL Port")
	dbPurgeCmd.Flags().StringVarP(&mysqlUser, "mysql-user", "u", "dap_oss", "MySQL Username")
	dbPurgeCmd.Flags().StringVarP(&mysqlPassword, "mysql-password", "P", "YzxdCT3S$%IDgpchDr", "MySQL Password")
	dbPurgeCmd.Flags().IntVarP(&purgeBatchSize, "batch-size", "b", 2000, "Batch size for deletion")
}
