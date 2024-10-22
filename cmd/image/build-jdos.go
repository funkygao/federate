package image

import (
	"log"

	"github.com/spf13/cobra"
)

var jdosCmd = &cobra.Command{
	Use:   "build-jdos",
	Short: "Build target system for deployment on JDOS 3.0",
	Run: func(cmd *cobra.Command, args []string) {
		buildForJDOS()
	},
}

func buildForJDOS() {
	log.Println("构建方式:")
	log.Println("   代码构建")
	log.Println("编译命令:")
	log.Println("   make jdos-stock-synergy-test")
	log.Println("制品路径(抽包地址):")
	log.Println("   wms-microfusion/generated/wms-stock-synergy/target")
	log.Println("基础镜像:")
	log.Println("   java-jd-centos7-jdk8.0.192-tom8.5.42-ngx197-emqx4.4.19-admin:latest")
	log.Println("submodule的同步方式:")
	log.Println("   锁定版本")
}
