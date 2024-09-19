package onpremise

var (
	// totp
	totpSecret = "T3GSIG8LwQHnIDd4FtlQPthQlp"
	totpTtlSec = 2 * 60

	// purge
	mysqlUser, mysqlPassword, mysqlHost, mysqlPort string
	purgeBatchSize                                 int
)
