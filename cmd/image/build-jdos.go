package image

import (
	"federate/internal/fs"
	"federate/pkg/manifest"
	"github.com/spf13/cobra"
)

var jdosCmd = &cobra.Command{
	Use:   "build-jdos",
	Short: "Generate target system Dockerfile for JDOS 3.0",
	Run: func(cmd *cobra.Command, args []string) {
		m := manifest.Load()
		doGenerateJdosDockerfile(m)
	},
}

func GenerateJdosDockerfile(m *manifest.Manifest, outfile string) (overwrite bool) {
	data := struct{}{}
	return fs.GenerateFileFromTmpl("templates/image/Dockerfile.jdos", outfile, data)
}

func doGenerateJdosDockerfile(m *manifest.Manifest) {
	GenerateJdosDockerfile(m, "")
}

func init() {
	manifest.RequiredManifestFileFlag(jdosCmd)
}
