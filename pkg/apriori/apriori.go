package apriori

import (
	"fmt"
	"path/filepath"
)

// ItemSet represents a set of items
type ItemSet map[string]struct{}

// Transaction represents a single transaction
type Transaction []string

// Association represents a file association
type Association struct {
	Items      [2]string
	Confidence float64
}

// Result represents the result of the Apriori algorithm
type Result struct {
	FrequentItemsets map[string]int
	Associations     []Association
}

// Run executes the Apriori algorithm on the given transactions
func Run(transactions []Transaction, minSupport, minConfidence float64, baseName bool) Result {
	itemsets := make(map[string]int)
	for _, transaction := range transactions {
		for _, item := range transaction {
			itemsets[item]++
		}
	}

	frequentItemsets := make(map[string]int)
	for item, count := range itemsets {
		support := float64(count) / float64(len(transactions))
		if support >= minSupport {
			frequentItemsets[item] = count
		}
	}

	associationMap := make(map[string]Association)
	for i := range frequentItemsets {
		for j := range frequentItemsets {
			if i != j {
				pairCount := 0
				for _, transaction := range transactions {
					if contains(transaction, i) && contains(transaction, j) {
						pairCount++
					}
				}
				confidence := float64(pairCount) / float64(frequentItemsets[i])
				if confidence >= minConfidence {
					if baseName {
						i = filepath.Base(i)
						j = filepath.Base(j)
					}
					key := getAssociationKey(i, j)
					newAssoc := Association{
						Items:      [2]string{i, j},
						Confidence: confidence,
					}
					if existingAssoc, exists := associationMap[key]; !exists || newAssoc.Confidence > existingAssoc.Confidence {
						associationMap[key] = newAssoc
					}
				}
			}
		}
	}

	associations := make([]Association, 0, len(associationMap))
	for _, assoc := range associationMap {
		associations = append(associations, assoc)
	}

	return Result{
		FrequentItemsets: frequentItemsets,
		Associations:     associations,
	}
}

func contains(slice []string, item string) bool {
	for _, a := range slice {
		if a == item {
			return true
		}
	}
	return false
}

func getAssociationKey(a, b string) string {
	if a < b {
		return fmt.Sprintf("%s <-> %s", a, b)
	}
	return fmt.Sprintf("%s <-> %s", b, a)
}
