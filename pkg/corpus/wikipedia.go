package corpus

import (
	"compress/bzip2"
	"io"
	"log"
	"os"
	"strings"

	"github.com/dustin/go-wikiparse"
)

func GetSimplewikiCorpus(bzFile string) string {
	f, err := os.Open(bzFile)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	// 解压 bzip2 文件
	bzr := bzip2.NewReader(f)

	// 解析 Wikipedia XML
	parser, err := wikiparse.NewParser(bzr)
	if err != nil {
		panic(err)
	}

	var corpusData strings.Builder
	articleCount := 0
	for {
		page, err := parser.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			panic(err)
		}
		// 检查是否为文章页面（排除重定向、分类等特殊页面）
		if page.Ns == 0 && !strings.HasPrefix(page.Title, "Wikipedia:") {
			corpusData.WriteString(page.Revisions[0].Text)
			corpusData.WriteString("\n")
			articleCount++
		}
	}
	log.Printf("%d articles read", articleCount)
	return corpusData.String()
}
