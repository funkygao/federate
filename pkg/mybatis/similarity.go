package mybatis

import (
	"regexp"
	"sort"
	"strings"
)

type SimilarityPair struct {
	ID1        string
	ID2        string
	Similarity float64
}

func (sa *SQLAnalyzer) ComputeSimilarities(topN int) map[string]map[string][]SimilarityPair {
	result := make(map[string]map[string][]SimilarityPair) // map[SQLType]map[Filename][]SimilarityPair

	for sqlType, statements := range sa.StatementsByType {
		// 初始化内层映射
		if _, ok := result[sqlType]; !ok {
			result[sqlType] = make(map[string][]SimilarityPair)
		}

		// 按文件名分组语句
		statementsByFile := make(map[string][]*Statement)
		for _, stmt := range statements {
			filename := stmt.Filename
			statementsByFile[filename] = append(statementsByFile[filename], stmt)
		}

		// 在每个文件中计算相似度
		for filename, stmtsInFile := range statementsByFile {
			var pairs []SimilarityPair
			// 对文件中的所有语句进行标记化和规范化
			tokensList := make([][]string, len(stmtsInFile))
			for i, stmt := range stmtsInFile {
				tokensList[i] = tokenizeAndNormalize(stmt.SQL)
			}

			// 计算两两之间的相似度
			for i := 0; i < len(stmtsInFile); i++ {
				for j := i + 1; j < len(stmtsInFile); j++ {
					similarity := computeJaccardSimilarity(tokensList[i], tokensList[j])
					pair := SimilarityPair{
						ID1:        stmtsInFile[i].ID,
						ID2:        stmtsInFile[j].ID,
						Similarity: similarity,
					}
					pairs = append(pairs, pair)
				}
			}

			// 按相似度排序
			sort.Slice(pairs, func(i, j int) bool {
				return pairs[i].Similarity > pairs[j].Similarity
			})

			// 每个XML文件内，选择前 N 个相似度最高的
			if len(pairs) > topN {
				pairs = pairs[:topN]
			}

			// 将结果存入返回值
			result[sqlType][filename] = pairs
		}
	}

	return result
}

func tokenizeAndNormalize(sql string) []string {
	// Remove literals and placeholders
	sql = regexp.MustCompile(`'[^']*'|\?`).ReplaceAllString(sql, "")
	// Convert to lowercase
	sql = strings.ToLower(sql)
	// Split into tokens (simple split by whitespace and punctuation)
	tokens := regexp.MustCompile(`\W+`).Split(sql, -1)
	// Remove empty tokens
	var normalizedTokens []string
	for _, token := range tokens {
		if token != "" {
			normalizedTokens = append(normalizedTokens, token)
		}
	}
	return normalizedTokens
}

func computeJaccardSimilarity(tokens1, tokens2 []string) float64 {
	set1 := make(map[string]struct{})
	set2 := make(map[string]struct{})
	for _, token := range tokens1 {
		set1[token] = struct{}{}
	}
	for _, token := range tokens2 {
		set2[token] = struct{}{}
	}
	var intersectionSize int
	for token := range set1 {
		if _, exists := set2[token]; exists {
			intersectionSize++
		}
	}
	unionSize := len(set1) + len(set2) - intersectionSize
	if unionSize == 0 {
		return 0
	}
	return float64(intersectionSize) / float64(unionSize)
}
