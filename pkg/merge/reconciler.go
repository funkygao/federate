package merge

type Reconciler interface {
	Reconcile(dryRun bool) error
}
