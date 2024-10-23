package image

import (
	"log"
	"os"
	"os/exec"
	"time"

	"federate/internal/fs"
	"github.com/spf13/cobra"
)

var profile string

var buildDockerCmd = &cobra.Command{
	Use:   "build-docker",
	Short: "Build app into Docker image to push to jdcloud image registry",
	Long: `Build app into Docker image to push to jdcloud image registry.

Example usage:
  federate image build-docker --image-repo wms-outbound --app-source-path wms-outbound-web-1.0.0-SNAPSHOT-package`,
	Run: func(cmd *cobra.Command, args []string) {
		buildDockerImage()
	},
}

func buildDockerImage() {
	if appName == "" || appSourcePath == "" {
		log.Fatalf("所有必需参数都必须提供")
	}

	// 记录开始时间
	startTime := time.Now()

	// 生成中间文件
	fs.GenerateFileFromTmpl("templates/image/Dockerfile.jdcloud", ".Dockerfile", map[string]string{
		"APP_SOURCE_PATH": appSourcePath,
	})
	fs.GenerateFileFromTmpl("templates/image/docker-deploy.sh", ".docker-deploy.sh", map[string]string{
		"PROFILE":    profile,
		"IMAGE_REPO": appName,
	})

	cmd := exec.Command("bash", ".docker-deploy.sh")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		log.Fatalf("Docker build 失败: %v", err)
	}

	// safe to delete tmp files after docker image built
	os.Remove(".Dockerfile")
	os.Remove(".docker-deploy.sh")

	// 计算并输出耗时
	elapsedTime := time.Since(startTime)
	log.Printf("操作耗时: %s\n", elapsedTime)
}

func init() {
	buildDockerCmd.Flags().StringVarP(&appName, "image-repo", "o", "", "镜像仓库名称")
	buildDockerCmd.Flags().StringVarP(&appSourcePath, "app-source-path", "s", "", "应用打包成品路径")
	buildDockerCmd.Flags().StringVarP(&profile, "profile", "p", "on-premise", "profile")
}
