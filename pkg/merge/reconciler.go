package merge

type Reconciler interface {
	Name() string
	Reconcile(dryRun bool) error
}
