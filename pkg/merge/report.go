package merge

type Reporter interface {
	Record(data interface{})
	ShowReport()
}
