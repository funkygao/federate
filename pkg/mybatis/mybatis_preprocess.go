package mybatis

import (
	"strings"
)

func (mp *MyBatisProcessor) processIncludes(sql string) string {
	for includeRegex.MatchString(sql) {
		sql = includeRegex.ReplaceAllStringFunc(sql, func(match string) string {
			refID := includeRegex.FindStringSubmatch(match)[1]
			return mp.processIncludes(mp.fragments[refID])
		})
	}
	return sql
}

func (mp *MyBatisProcessor) preprocessRawSQL(rawSQL string) string {
	// 处理 <include> 标签
	rawSQL = mp.processIncludes(rawSQL)

	// 处理 <where> 标签
	rawSQL = whereRegex.ReplaceAllStringFunc(rawSQL, func(match string) string {
		inner := whereRegex.FindStringSubmatch(match)[1]
		return "WHERE " + strings.TrimSpace(inner)
	})

	// 处理 <choose> 标签
	rawSQL = chooseRegex.ReplaceAllStringFunc(rawSQL, func(match string) string {
		whenMatches := whenRegex.FindAllStringSubmatch(match, -1)
		for _, whenMatch := range whenMatches {
			if len(whenMatch) > 1 {
				return "?"
			}
		}
		if otherwiseMatch := otherwiseRegex.FindStringSubmatch(match); len(otherwiseMatch) > 1 {
			return strings.TrimSpace(otherwiseMatch[1])
		}
		return "?"
	})

	// 处理 <if> 标签
	rawSQL = ifRegex.ReplaceAllString(rawSQL, "$1")

	// 处理 <foreach> 标签，特别处理批量插入
	rawSQL = foreachRegex.ReplaceAllStringFunc(rawSQL, func(match string) string {
		innerContent := foreachRegex.FindStringSubmatch(match)[1]
		if strings.Contains(strings.ToLower(rawSQL), "insert into") && strings.Contains(strings.ToLower(rawSQL), "values") {
			// 批量插入情况
			valueCount := mp.countInsertPlaceholders(innerContent)
			placeholders := make([]string, valueCount)
			for i := range placeholders {
				placeholders[i] = "?"
			}
			return "(" + strings.Join(placeholders, ", ") + "), (" + strings.Join(placeholders, ", ") + ")"
		}
		if strings.Contains(innerContent, "->>") {
			// JSON 操作符情况
			return "(" + jsonOperatorRegex.ReplaceAllString(innerContent, "$1 ->> ? = ?") + ")"
		}
		return "(?)"
	})

	// 替换变量，保留 JSON 操作符
	rawSQL = jsonOperatorRegex.ReplaceAllString(rawSQL, "$1 ->> ? = ?")
	rawSQL = dollarVarRegex.ReplaceAllString(rawSQL, "?")
	rawSQL = hashVarRegex.ReplaceAllString(rawSQL, "?")

	// 移除所有剩余的 XML 标签
	rawSQL = tagRegex.ReplaceAllString(rawSQL, "")

	// 清理多余的空白字符
	rawSQL = strings.Join(strings.Fields(rawSQL), " ")

	return strings.TrimSpace(rawSQL)
}

func (mp *MyBatisProcessor) countInsertPlaceholders(content string) int {
	count := 0

	// 计算 #{...} 的数量
	count += strings.Count(content, "#{")

	// 计算显式的 ? 占位符
	count += strings.Count(content, "?")

	// 计算数字字面量（如 0）的数量
	count += len(numberRegex.FindAllString(content, -1))

	return count
}
