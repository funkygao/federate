package cmd

import (
	"federate/pkg/tablerender"
	"github.com/spf13/cobra"
)

var jdosCmd = &cobra.Command{
	Use:   "jdos",
	Short: "How to config JDOS 3.0 CI/CD for fusion project",
	Run: func(cmd *cobra.Command, args []string) {
		setupJDOS()
	},
}

func setupJDOS() {
	var configs = [][]string{
		{"成员管理", "JDOSBOOT", "融合代码库需要为该用户分配 Guest 权限"},
		{"打包类型", "maven-3", ""},
		{"编译语言", "java-8", ""},
		{"基础镜像", "base_worker/java-jd-centos7-jdk8.0.192-tom8.5.42-ngx197:latest", "包含了 make/java/maven"},
		{"编译命令", "bin/federate microservice scaffold\nmake fusion-start\nmake make consolidate ENV=<env>\nmvn validate -q", "JDOS会在最后一行上添加 '-f $(pwd) -T 1C -Dmaven.artifact.threads=16'"},
	}
	header := []string{"Item", "Config Value", "Remark"}
	tablerender.DisplayTable(header, configs, false)
}
