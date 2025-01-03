package insight

import (
	"os"
	"path/filepath"
	"strings"

	"federate/pkg/java"
	"federate/pkg/mybatis"
	"federate/pkg/primitive"
	"github.com/spf13/cobra"
)

var mybatisCmd = &cobra.Command{
	Use:   "mybatis <dir>",
	Short: "Analyze MyBatis MySQL mapper XML files in the specified directory",
	Args:  cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		dir := args[0]
		analyzeMybatisMapperXML(dir)
	},
}

func analyzeMybatisMapperXML(dir string) {
	analyzer := mybatis.NewAnalyzer([]string{"deleted", "create_time", "yn"})

	var files []string
	fileChan, _ := java.ListFilesAsync_(dir, java.IsXML, walkDir)
	for f := range fileChan {
		files = append(files, f.Path)
	}

	analyzer.AnalyzeFiles(files)
	analyzer.GenerateReport()
}

var skippedDirs = map[string]struct{}{
	"target": primitive.Empty,
	"test":   primitive.Empty,
}

func walkDir(info os.FileInfo) error {
	name := info.Name()

	if _, shouldSkip := skippedDirs[name]; shouldSkip {
		return filepath.SkipDir
	}

	if len(name) > 2 && strings.HasPrefix(name, ".") { // .git, .idea
		return filepath.SkipDir
	}

	return nil
}

func init() {
	mybatisCmd.Flags().IntVarP(&mybatis.TopK, "top", "t", 20, "Number of top elements to display in bar chart")
	mybatisCmd.Flags().IntVarP(&mybatis.Verbosity, "verbosity", "v", 1, "Ouput verbosity level: 1-5")
	mybatisCmd.Flags().Float64VarP(&mybatis.SimilarityThreshold, "similarity-threshold", "s", 0.75, "Statement similary threhold")
}
