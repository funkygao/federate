package mybatis

import (
	"regexp"

	"github.com/beevik/etree"
)

type SQLFragments map[string]string

var (
	RootTag = "mapper"

	includeRegex   = regexp.MustCompile(`<include\s+refid="([^"]+)"(?:\s*/>|[^>]*>\s*</include>)`)
	whereRegex     = regexp.MustCompile(`<where>(?s)(.*?)</where>`)
	chooseRegex    = regexp.MustCompile(`<choose>(?s)(.*?)</choose>`)
	whenRegex      = regexp.MustCompile(`<when[^>]*>(?s)(.*?)</when>`)
	ifRegex        = regexp.MustCompile(`<if[^>]*>(?s)(.*?)</if>`)
	otherwiseRegex = regexp.MustCompile(`<otherwise>(?s)(.*?)</otherwise>`)
	foreachRegex   = regexp.MustCompile(`<foreach[^>]*>(?s)(.*?)</foreach>`)
	dollarVarRegex = regexp.MustCompile(`\$\{([^}]+)\}`)
	hashVarRegex   = regexp.MustCompile(`#\{([^}]+)\}`)
	tagRegex       = regexp.MustCompile(`</?[^>]+>`)
	numberRegex    = regexp.MustCompile(`\b\d+\b`)

	jsonOperatorRegex      = regexp.MustCompile(`(\w+)\s*->>\s*#\{[^}]+\}\s*=\s*#\{[^}]+\}`)
	variableInForeachRegex = regexp.MustCompile(`#\{[^}]+\}`)
)

type MyBatisProcessor struct {
	fragments SQLFragments
}

func NewMyBatisProcessor() *MyBatisProcessor {
	return &MyBatisProcessor{fragments: make(SQLFragments)}
}

func (mp *MyBatisProcessor) PreprocessStmt(stmt *etree.Element) (rawSQL, preprocessedSQL, stmtID string) {
	rawSQL = mp.extractRawSQL(stmt)
	preprocessedSQL = mp.preprocessRawSQL(rawSQL)
	stmtID = stmt.SelectAttrValue("id", "")
	return
}
