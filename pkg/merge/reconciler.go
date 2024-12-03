package merge

type Reconciler interface {
	Name() string

	Reconcile() error
}

type Preparer interface {
	Reconciler
	Prepare() error
}
