// Apriori算法是一种用于关联规则挖掘的经典算法：应用于推荐
// 主要优点是其简单性和易于理解，但在处理大型数据集时可能会面临效率问题，特别是当存在大量频繁项集或者很长的事务时
package main

import (
	"fmt"
	"sort"
	"strings"
)

// Transaction 表示一次购物中买的所有商品
// 例如，一个顾客在超市一次购物买了面包、牛奶和鸡蛋，
// 那么这次购物就可以表示为 ["面包", "牛奶", "鸡蛋"]
type Transaction []string

// ItemSet 表示一组经常一起购买的商品
// 例如，{牛奶, 面包} 表示牛奶和面包经常被一起购买
type ItemSet map[string]struct{}

// FrequentItemSet 表示一个频繁项集，包含一组商品及其在所有购物中出现的次数
// 例如，{商品: {牛奶, 面包}, 出现次数: 100} 表示牛奶和面包一起被购买了100次
type FrequentItemSet struct {
	Items   ItemSet
	Support int
}

// generateCandidates 生成可能频繁出现的商品组合
// 这个函数基于已知的频繁商品组合，尝试构建更大的组合
// 例如，如果 {牛奶, 面包} 和 {牛奶, 鸡蛋} 都是频繁的，
// 那么我们会尝试生成 {牛奶, 面包, 鸡蛋} 作为新的候选
func generateCandidates(level int, prevFreqSets []FrequentItemSet) []ItemSet {
	candidates := []ItemSet{}

	// 使用 Apriori 原理：频繁项集的所有非空子集也必须是频繁的
	for i := 0; i < len(prevFreqSets); i++ {
		for j := i + 1; j < len(prevFreqSets); j++ {
			set1 := prevFreqSets[i].Items
			set2 := prevFreqSets[j].Items

			// 尝试合并两个项集
			newSet := make(ItemSet)
			for item := range set1 {
				newSet[item] = struct{}{}
			}

			canJoin := true
			for item := range set2 {
				if _, exists := newSet[item]; !exists {
					if len(newSet) == level-1 {
						newSet[item] = struct{}{}
					} else {
						canJoin = false
						break
					}
				}
			}

			// 如果可以合并且新项集大小正确，则添加到候选集
			if canJoin && len(newSet) == level {
				candidates = append(candidates, newSet)
			}
		}
	}

	return candidates
}

// calculateSupport 计算一组商品在所有购物记录中一起出现的次数
// 这个次数就是该商品组合的"支持度"
func calculateSupport(itemSet ItemSet, transactions []Transaction) int {
	count := 0
	for _, transaction := range transactions {
		contains := true
		for item := range itemSet {
			if !containsItem(transaction, item) {
				contains = false
				break
			}
		}
		if contains {
			count++
		}
	}
	return count
}

// containsItem 检查一次购物中是否包含特定商品
func containsItem(transaction Transaction, item string) bool {
	for _, i := range transaction {
		if i == item {
			return true
		}
	}
	return false
}

// apriori 实现 Apriori 算法的主要逻辑
// Apriori 算法的核心思想是：
// 1. 从单个商品开始，逐步构建更大的商品组合
// 2. 只保留频繁出现的商品组合（出现次数达到最小支持度）
// 3. 利用"如果一个商品组合是频繁的，它的所有子集也一定是频繁的"这个性质来优化搜索过程
func apriori(transactions []Transaction, minSupport int) []FrequentItemSet {
	// 第一步：找出所有的单个项目
	allItems := make(ItemSet)
	for _, transaction := range transactions {
		for _, item := range transaction {
			allItems[item] = struct{}{}
		}
	}

	level := 1
	var frequentItemSets []FrequentItemSet
	frequentItemSetsMap := make(map[string]FrequentItemSet)

	// 迭代构建频繁项集，从单个商品开始，逐步增加组合中的商品数量
	for {
		var candidates []ItemSet
		if level == 1 {
			// 对于第一轮，候选项就是所有单个商品
			for item := range allItems {
				candidates = append(candidates, ItemSet{item: struct{}{}})
			}
		} else {
			// 生成可能的频繁商品组合
			candidates = generateCandidates(level, frequentItemSets)
		}

		// 如果没有候选项，算法结束
		if len(candidates) == 0 {
			break
		}

		// 计算每个候选商品组合的出现次数，保留频繁出现的组合
		for _, candidate := range candidates {
			support := calculateSupport(candidate, transactions)
			if support >= minSupport {
				key := itemSetToString(candidate)
				if _, exists := frequentItemSetsMap[key]; !exists {
					fis := FrequentItemSet{Items: candidate, Support: support}
					frequentItemSets = append(frequentItemSets, fis)
					frequentItemSetsMap[key] = fis
				}
			}
		}

		level++
	}

	return frequentItemSets
}

// 将ItemSet转换为字符串，用于去重
func itemSetToString(itemSet ItemSet) string {
	items := make([]string, 0, len(itemSet))
	for item := range itemSet {
		items = append(items, item)
	}
	sort.Strings(items)
	return strings.Join(items, ",")
}

func main() {
	// 示例购物数据
	transactions := []Transaction{
		{"A", "B", "C", "D"},
		{"B", "C", "E"},
		{"A", "B", "C", "E"},
		{"B", "D", "E"},
		{"A", "B", "C", "D"},
	}

	minSupport := 2 // 最小支持度：一个商品组合至少要在2次购物中出现才被认为是"频繁的"

	frequentItemSets := apriori(transactions, minSupport)

	// 对频繁项集进行排序
	sort.Slice(frequentItemSets, func(i, j int) bool {
		if len(frequentItemSets[i].Items) != len(frequentItemSets[j].Items) {
			return len(frequentItemSets[i].Items) < len(frequentItemSets[j].Items)
		}
		return frequentItemSets[i].Support > frequentItemSets[j].Support
	})

	fmt.Println("频繁购买的商品组合:")
	for _, fis := range frequentItemSets {
		items := make([]string, 0, len(fis.Items))
		for item := range fis.Items {
			items = append(items, item)
		}
		sort.Strings(items)
		fmt.Printf("%v (出现次数: %d)\n", items, fis.Support)
	}
}
