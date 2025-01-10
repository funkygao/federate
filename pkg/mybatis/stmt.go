package mybatis

import (
	"fmt"
	"strings"

	"github.com/beevik/etree"
	"github.com/xwb1989/sqlparser"
)

type SQLet struct {
	SQL     string
	SQLType string
	Stmt    sqlparser.Statement
	Primary bool
}

type Statement struct {
	Filename      string
	ID            string
	Tag           string // XML Tag
	ResultType    string
	ParameterType string

	XMLText string
	Timeout int

	Metadata   StatementMetadata
	Complexity CognitiveComplexity

	// 可能是多个语句组成的：SET @affected_rows = 0; UDPATE ...; set @affected_rows  = @affected_rows + row_count(); select @affected_rows as rows;
	SQL string

	SubSQL []SQLet
}

func (s *Statement) Analyze() error {
	parseErrors := s.parseSQL()
	if len(parseErrors) > 0 {
		return fmt.Errorf("encountered %d parse errors: %v", len(parseErrors), parseErrors)
	}

	s.analyzeMetadata()
	s.analyzeCognitiveComplexity()

	return nil
}

func (s *Statement) MinimalSQL() string {
	var result strings.Builder
	var inString, inLineComment, inBlockComment bool
	var stringDelimiter rune

	for i := 0; i < len(s.SQL); i++ {
		char := rune(s.SQL[i])

		switch {
		case inString:
			result.WriteRune(char)
			if char == stringDelimiter && (i == 0 || s.SQL[i-1] != '\\') {
				inString = false
			}
		case inLineComment:
			if char == '\n' {
				inLineComment = false
				result.WriteRune(' ')
			}
		case inBlockComment:
			if char == '*' && i+1 < len(s.SQL) && s.SQL[i+1] == '/' {
				inBlockComment = false
				i++
				result.WriteRune(' ')
			}
		case char == '\'' || char == '"':
			inString = true
			stringDelimiter = char
			result.WriteRune(char)
		case char == '-' && i+1 < len(s.SQL) && s.SQL[i+1] == '-':
			inLineComment = true
			i++
		case char == '/' && i+1 < len(s.SQL) && s.SQL[i+1] == '*':
			inBlockComment = true
			i++
		case char == '\n' || char == '\t':
			result.WriteRune(' ')
		default:
			result.WriteRune(char)
		}
	}

	// 移除多余的空格
	return strings.Join(strings.Fields(result.String()), " ")
}

func (s *Statement) PrimaryASTs() []sqlparser.Statement {
	var result []sqlparser.Statement
	for _, sqlet := range s.SubSQL {
		if sqlet.Primary {
			result = append(result, sqlet.Stmt)
		}
	}
	return result
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
	if err := doc.ReadFromString(s.XMLText); err != nil {
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

func (s *Statement) HasOnDuplicateKey() bool {
	return strings.Contains(strings.ToUpper(s.SQL), "ON DUPLICATE KEY")
}

func (s *Statement) HasPessimisticLocking() bool {
	return strings.Contains(strings.ToUpper(s.SQL), "FOR UPDATE") ||
		strings.Contains(strings.ToUpper(s.SQL), "LOCK IN SHARE MODE")
}

func (s *Statement) HasOptimisticLocking() bool {
	if s.Tag == "update" {
		if strings.Contains(s.SQL, "version = version + 1") {
			return true
		}
	}

	return false
}

func (s *Statement) addSubSQL(sqlet SQLet) {
	s.SubSQL = append(s.SubSQL, sqlet)
}

func (s *Statement) SubN() int {
	return len(s.SubSQL)
}

func (s *Statement) splitSQL() []string {
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
