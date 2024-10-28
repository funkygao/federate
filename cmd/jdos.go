package cmd

import (
	"log"

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
	log.Println("成员管理：JDOSBOOT")
	log.Println("打包类型：maven-3")
	log.Println("编译语言：java-8")
	log.Println("基础镜像：base_worker/java-jd-centos7-jdk8.0.192-tom8.5.42-ngx197:latest")
	log.Println("编译命令：")
	log.Println("    bin/federate microservice scaffold")
	log.Println("    make fusion-start")
	log.Println("    make make consolidate ENV=<env>")
	log.Println("    mvn validate -q # JDOS会在最后一行上添加 '-f $(pwd) -T 1C -Dmaven.artifact.threads=16'")
}
