package mybatis

import (
	"strings"

	"github.com/beevik/etree"
)

type Statement struct {
	Filename string
	ID       string
	Type     string
	Raw      string
	SQL      string
	Timeout  int
}

func (s *Statement) IsBatchOperation() bool {
	containsSQLOperation := func(elem *etree.Element) bool {
		text := strings.ToUpper(elem.Text())
		return strings.Contains(text, "INSERT") ||
			strings.Contains(text, "UPDATE") ||
			strings.Contains(text, "DELETE") ||
			strings.Contains(text, "SELECT")
	}

	doc := etree.NewDocument()
	if err := doc.ReadFromString(s.Raw); err != nil {
		return false
	}

	root := doc.Root()
	if root == nil {
		return false
	}

	// 检查是否有最外层的 foreach
	for _, child := range root.ChildElements() {
		if child.Tag == "foreach" && containsSQLOperation(child) {
			return true
		}
	}

	return false
}

func (s *Statement) SplitSQL() []string {
	var statements []string
	var currentStmt strings.Builder
	var inString, inComment bool
	var stringDelimiter rune

	for i, char := range s.SQL {
		switch {
		case inString:
			currentStmt.WriteRune(char)
			if char == stringDelimiter && s.SQL[i-1] != '\\' {
				inString = false
			}
		case inComment:
			if char == '\n' {
				inComment = false
			}
			currentStmt.WriteRune(char)
		case char == '\'' || char == '"':
			inString = true
			stringDelimiter = char
			currentStmt.WriteRune(char)
		case char == '-' && i+1 < len(s.SQL) && s.SQL[i+1] == '-':
			inComment = true
			currentStmt.WriteRune(char)
		case char == ';' && !inString && !inComment:
			stmt := strings.TrimSpace(currentStmt.String())
			if stmt != "" {
				statements = append(statements, stmt)
			}
			currentStmt.Reset()
		default:
			currentStmt.WriteRune(char)
		}
	}

	if currentStmt.Len() > 0 {
		stmt := strings.TrimSpace(currentStmt.String())
		if stmt != "" {
			statements = append(statements, stmt)
		}
	}

	return statements
}
