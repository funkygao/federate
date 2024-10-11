package microservice

import (
	"fmt"
	"os"

	"federate/internal/fs"
	"github.com/alecthomas/chroma/formatters"
	"github.com/alecthomas/chroma/lexers"
	"github.com/alecthomas/chroma/styles"
	"github.com/spf13/cobra"
)

var manifestCmd = &cobra.Command{
	Use:   "manifest",
	Short: "Explain the manifest.yaml",
	Long:  `The manifest command explains the manifest.yaml.`,
	Run: func(cmd *cobra.Command, args []string) {
		showManifest()
	},
}

func showManifest() {
	yaml, _ := fs.FS.ReadFile("templates/manifest.yaml")
	lexer := lexers.Get("yaml")
	iterator, err := lexer.Tokenise(nil, string(yaml))
	if err != nil {
		fmt.Println("Error:", err)
		return
	}

	style := styles.Get("pygments")
	formatter := formatters.Get("terminal")
	formatter.Format(os.Stdout, style, iterator)
}
