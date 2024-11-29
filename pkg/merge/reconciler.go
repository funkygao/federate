package merge

type Reconciler interface {
	Name() string

	Reconcile() error
}
