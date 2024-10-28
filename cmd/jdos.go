package cmd

import (
	"fmt"
	"log"
	"os"

	"federate/internal/fs"
	"federate/pkg/manifest"
	"federate/pkg/tablerender"
	"github.com/alecthomas/chroma/formatters"
	"github.com/alecthomas/chroma/lexers"
	"github.com/alecthomas/chroma/styles"
	"github.com/spf13/cobra"
)

var showDocker bool

var jdosCmd = &cobra.Command{
	Use:   "jdos",
	Short: "Configure JDOS 3.0 CI/CD for fusion project",
	Run: func(cmd *cobra.Command, args []string) {
		if showDocker {
			displayDocker()
		} else {
			manifest := manifest.Load()
			setupJDOS(manifest)
		}
	},
}

func setupJDOS(m *manifest.Manifest) {
	var configs = [][]string{
		{"构建方式", "代码构建", "源码地址：填写你的融合项目代码库"},
		{"成员管理", "JDOSBOOT", "融合代码库需要为该用户分配 Guest 权限"},
		{"基础镜像", "base_worker/java-jd-centos7-jdk8.0.192-tom8.5.42-ngx197:latest", "包含了 make/java/maven"},
	}
	header := []string{"Item", "Config Value", "Remark"}
	tablerender.DisplayTable(header, configs, true)
	log.Println("编译命令:")
	data := struct {
		Name string
	}{
		Name: m.Main.Name,
	}
	fs.GenerateFileFromTmpl("templates/jdos.compile.sh", "", data)
}

func displayDocker() {
	data, _ := fs.FS.ReadFile("templates/image/Dockerfile.jdos.builtin")
	lexer := lexers.Get("docker")
	iterator, err := lexer.Tokenise(nil, string(data))
	if err != nil {
		fmt.Println("Error:", err)
		return
	}

	style := styles.Get("pygments")
	formatter := formatters.Get("terminal")
	formatter.Format(os.Stdout, style, iterator)
}

func init() {
	manifest.RequiredManifestFileFlag(jdosCmd)
	jdosCmd.Flags().BoolVarP(&showDocker, "dockerfile", "d", false, "Display JDOS generated Dockerfile")
}
