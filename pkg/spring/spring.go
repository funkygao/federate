package spring

type SearchType int

const (
	SearchByID SearchType = iota
	SearchByRef
)

// BeanInfo 结构体用于存储 bean 的信息
type BeanInfo struct {
	// Identifier 可能是 bean 的 ID/name（当 Type 为 SearchByID 时），
	// 或者是 ref 值（当 Type 为 SearchByRef 时）
	Identifier string

	FileName string
}

type SpringManager interface {
	ListBeans(springXmlPath string, searchType SearchType) []BeanInfo

	ChangeBeans(springXmlPath string, searchType SearchType, updateMap map[string]string) error
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
