package debug

import (
	"log"
	"path/filepath"
	"sort"

	"federate/pkg/manifest"
	"federate/pkg/spring"
	"github.com/spf13/cobra"
)

var showConflictOnly bool

var refCmd = &cobra.Command{
	Use:   "ref",
	Short: "List bean ref values from federated/spring.xml",
	Run: func(cmd *cobra.Command, args []string) {
		m := manifest.Load()
		listRef(m)
	},
}

func listRef(m *manifest.Manifest) {
	manager := spring.New(true)
	springXmlPath := filepath.Join(m.TargetResourceDir(), "federated/spring.xml")
	refs := make(map[string]map[string]struct{})

	for _, bean := range manager.ListBeans(springXmlPath, spring.SearchByRef) {
		if _, exists := refs[bean.Identifier]; !exists {
			refs[bean.Identifier] = make(map[string]struct{})
		}
		refs[bean.Identifier][bean.FileName] = struct{}{}
	}

	// 获取所有的 ref 值并排序
	refValues := make([]string, 0, len(refs))
	for ref, fileSet := range refs {
		if showConflictOnly && len(fileSet) < 2 {
			continue
		}
		refValues = append(refValues, ref)
	}
	sort.Strings(refValues)

	// 输出排序后的结果
	for _, ref := range refValues {
		fileSet := refs[ref]
		files := make([]string, 0, len(fileSet))
		for file := range fileSet {
			files = append(files, file)
		}
		sort.Strings(files) // 对文件名进行排序

		log.Printf("Ref: %s\n", ref)
		for _, file := range files {
			log.Printf("  - %s\n", file)
		}
		log.Println() // 在每个 ref 之后添加一个空行，以提高可读性
	}
}

func init() {
	refCmd.Flags().BoolVarP(&showConflictOnly, "conflict-only", "c", true, "Display only possible conflict bean references")
}
