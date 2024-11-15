package spring

type SpringManager interface {
	SearchBean(springXmlPath string, beanId string) (found bool, file string)
}

type manager struct {
}

func New() SpringManager {
	return &manager{}
}
