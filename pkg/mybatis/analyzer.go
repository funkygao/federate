package mybatis

import (
	"log"

	"github.com/schollz/progressbar/v3"
)

var (
	Verbosity           int
	TopK                int
	SimilarityThreshold float64
	AnalyzeGit          bool
	Prompt              string
	GeneratePrompt      bool
	PromptSQL           bool
	ShowIndexRecommend  bool
	ShowBatchOps        bool
	ShowSimilarity      bool
)

type Analyzer struct {
	aggregator      *Aggregator
	reportGenerator *ReportGenerator
	mapperBuilders  map[string]*XMLMapperBuilder
}

func NewAnalyzer(ignoredFields []string) *Analyzer {
	a := &Analyzer{
		reportGenerator: NewReportGenerator(),
		mapperBuilders:  make(map[string]*XMLMapperBuilder),
	}
	a.aggregator = NewAggregator(a.mapperBuilders, ignoredFields)
	return a
}

func (a *Analyzer) AnalyzeFiles(files []string) {
	var bar *progressbar.ProgressBar
	if AnalyzeGit {
		bar = progressbar.NewOptions(len(files),
			progressbar.OptionEnableColorCodes(true),
			progressbar.OptionShowElapsedTimeOnFinish(),
			progressbar.OptionSetWidth(10),
			progressbar.OptionShowDescriptionAtLineEnd(),
			progressbar.OptionSetTheme(progressbar.Theme{
				Saucer:        "[green]■[reset]",
				SaucerHead:    "[green]►[reset]",
				SaucerPadding: "-",
				BarStart:      "[",
				BarEnd:        "]",
			}))
	}

	// prepare sql fragments
	for _, file := range files {
		if AnalyzeGit {
			bar.Add(1)
			bar.Describe(file)
		}

		if err := a.prepareFile(file); err != nil {
			log.Printf("%s %v", file, err)
		}
	}

	if AnalyzeGit {
		bar.Finish()
		log.Println()
		log.Println()
	}

	// analyze CRUD
	for _, file := range files {
		if err := a.analyzeFile(file); err != nil {
			log.Printf("%s %v", file, err)
		}
	}

	a.aggregator.Aggregate()
}

func (a *Analyzer) prepareFile(filePath string) error {
	xml := NewXMLMapperBuilder(filePath)
	if err := xml.Prepare(); err != nil && err != ErrNotMapperXML {
		return err
	}

	a.mapperBuilders[filePath] = xml
	return nil
}

func (a *Analyzer) analyzeFile(filePath string) error {
	// 解析 XML
	xml := a.mapperBuilders[filePath]
	if xml == nil {
		// not a mapper XML, e,g. pom.xml
		return nil
	}

	stmts, err := xml.Parse()
	if err != nil {
		if err == ErrNotMapperXML {
			return nil
		}
		return err
	}

	// <include refid=""/> 没有找到被引用 sql fragment
	a.aggregator.UnknownFragments[filePath] = xml.UnknownFragments

	// 解析 SQL AST
	for _, stmt := range stmts {
		a.aggregator.OnStmt(*stmt)
	}

	return nil
}

func (a *Analyzer) GenerateReport() {
	a.reportGenerator.Generate(a.aggregator)
}
