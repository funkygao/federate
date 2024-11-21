package conflict

type collector struct {
}

func NewManager() Collector {
	return &collector{}
}
