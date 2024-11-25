package transformer

import (
	"log"
)

var tf = &transformManager{}

func Get() Transformer {
	return tf
}

type Transformer interface {
	TransformTransaction(component, txManagerName string)

	ShowSummary()
}

type transformManager struct {
	transactions map[string]string
}

func (t *transformManager) TransformTransaction(component, txManagerName string) {
	t.transactions[component] = txManagerName
}

func (t *transformManager) ShowSummary() {
	for component, txManagerName := range t.transactions {
		log.Printf("%s uses transactionManager:%s", component, txManagerName)
	}
}
