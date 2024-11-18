package spring

type SpringManager interface {
	SearchBean(springXmlPath string, beanId string) (found bool, file string)
}

type manager struct {
	beanFullTags map[string]struct{}

	showUnregistered bool
	unregisteredTags map[string]struct{}
}

func New() SpringManager {
	return &manager{
		beanFullTags: map[string]struct{}{
			"bean":               struct{}{},
			"util:map":           struct{}{},
			"util:list":          struct{}{},
			"laf-config:manager": struct{}{},
			"jmq:producer":       struct{}{},
			"jmq:consumer":       struct{}{},
			"jmq:transport":      struct{}{},
			"jsf:consumer":       struct{}{},
			"jsf:consumerGroup":  struct{}{},
			"jsf:provider":       struct{}{},
			"jsf:filter":         struct{}{},
			"jsf:server":         struct{}{},
			"jsf:registry":       struct{}{},
			"dubbo:reference":    struct{}{},
			"dubbo:service":      struct{}{},
		},
		showUnregistered: false,
		unregisteredTags: make(map[string]struct{}),
	}
}
