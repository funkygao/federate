package cmd

import (
	"fmt"
	"log"
	"os"
	"os/exec"
	"time"

	"github.com/AlecAivazis/survey/v2"
	"github.com/spf13/cobra"
)

const jmxExporterPortBase = 10000

var (
	appName           string
	appSourcePath     string
	tomcatPort        int
	jvmSize           string
	cpuAffinity       string
	mainModule        string
	debugImage        bool
	defaultRpmRelease = time.Now().Format("20060102150405")
	rpmRelease        string
)

var buildRpmCmd = &cobra.Command{
	Use:   "build-rpm",
	Short: "Build app into rpm to upload to yum repo",
	Long: `Build app into rpm to upload to yum repo.

Example Makefile:

build-rpm:
	@federate image build-rpm --app-name wms-outbound --app-source-path wms-outbound-web/target/wms-outbound-web-1.0.0-SNAPSHOT-package --jvm-size large --main-module com.jdwl.wms.WebApplication --tomcat-port 8098`,
	Run: func(cmd *cobra.Command, args []string) {
		buildRPM()
	},
}

func buildRPM() {
	dockerImage := "rpm-" + appName
	if debugImage {
		if appName == "" {
			fmt.Println("--app-name must be provided")
			return
		}
		log.Printf(`docker run -it --rm --entrypoint /bin/bash %s`, dockerImage)
		return
	}
	// 定义需要提示的 survey 问题
	questions := []*survey.Question{}

	if appName == "" {
		questions = append(questions, &survey.Question{
			Name:     "appName",
			Prompt:   &survey.Input{Message: "请输入应用名称 (APP_NAME):"},
			Validate: survey.Required,
		})
	}

	if appSourcePath == "" {
		questions = append(questions, &survey.Question{
			Name:     "appSourcePath",
			Prompt:   &survey.Input{Message: "请输入应用打包成品路径，例如：wms-outbound-web/target/wms-outbound-web-1.0.0-SNAPSHOT-package (APP_SOURCE_PATH):"},
			Validate: survey.Required,
		})
	}

	if tomcatPort == 0 {
		questions = append(questions, &survey.Question{
			Name:     "tomcatPort",
			Prompt:   &survey.Input{Message: "请输入 TOMCAT 端口 (TOMCAT_PORT):"},
			Validate: survey.Required,
		})
	}

	if jvmSize == "" {
		questions = append(questions, &survey.Question{
			Name: "jvmSize",
			Prompt: &survey.Select{
				Message: "请输入 JVM 大小 (JVM_SIZE):",
				Options: []string{"large", "medium", "small"},
				Default: "medium",
			},
		})
	}

	if mainModule == "" {
		questions = append(questions, &survey.Question{
			Name:     "mainModule",
			Prompt:   &survey.Input{Message: "请输入 Main Module，例如：com.jdwl.wms.WebApplication (MAIN_MODULE):"},
			Validate: survey.Required,
		})
	}

	// 如果有需要提示的问题，则使用 survey 获取用户输入
	if len(questions) > 0 {
		answers := struct {
			AppName       string
			AppSourcePath string
			TomcatPort    int
			JvmSize       string
			CpuAffinity   string
			MainModule    string
		}{}

		err := survey.Ask(questions, &answers)
		if err != nil {
			log.Fatalf("无法获取输入: %v", err)
		}

		// 仅在未通过 flag 提供时更新变量
		if appName == "" {
			appName = answers.AppName
		}
		if appSourcePath == "" {
			appSourcePath = answers.AppSourcePath
		}
		if tomcatPort == 0 {
			tomcatPort = answers.TomcatPort
		}
		if jvmSize == "" {
			jvmSize = answers.JvmSize
		}
		if mainModule == "" {
			mainModule = answers.MainModule
		}
	}

	// 检查所有必需参数是否已提供
	if appName == "" || appSourcePath == "" || tomcatPort == 0 || jvmSize == "" || mainModule == "" {
		log.Fatalf("所有必需参数都必须提供: app-name, app-source-path, tomcat-port, jvm-size, main-module")
	}

	// 记录开始时间
	startTime := time.Now()

	jmxExporterPort := jmxExporterPortBase + tomcatPort

	// 生成中间文件
	generateFileFromTmpl("templates/Dockerfile.rpmbuilder", "Dockerfile.rpmbuilder", nil)
	generateFileFromTmpl("templates/config_rpm.sh", "config_rpm.sh", map[string]string{
		"TOMCAT_PORT":       fmt.Sprintf("%d", tomcatPort),
		"JMX_EXPORTER_PORT": fmt.Sprintf("%d", jmxExporterPort),
		"JVM_SIZE":          jvmSize,
		"CPU_AFFINITY":      cpuAffinity,
	})
	generateFileFromTmpl("templates/build_rpm.sh", "build_rpm.sh", nil)
	generateFileFromTmpl("templates/myapp.spec.template", "myapp.spec.template", nil)
	generateFileFromTmpl("templates/run.sh", "run.sh", map[string]string{
		"MAIN_MODULE": mainModule,
	})

	log.Println("正在构建 Docker 镜像...")
	dockerBuildCmd := exec.Command("docker", "buildx", "build", "--platform", "linux/amd64", "-f", "Dockerfile.rpmbuilder",
		"--build-arg", "APP_NAME="+appName,
		"--build-arg", "APP_SOURCE_PATH="+appSourcePath,
		"-t", dockerImage, "--load", ".")
	dockerBuildCmd.Stdout = os.Stdout
	dockerBuildCmd.Stderr = os.Stderr
	if err := dockerBuildCmd.Run(); err != nil {
		log.Fatalf("Docker build 失败: %v", err)
	}

	// Defer removal of temporary files
	tempFiles := []string{
		"Dockerfile.rpmbuilder",
		"config_rpm.sh",
		"build_rpm.sh",
		"myapp.spec.template",
		"run.sh",
	}
	defer func(files []string) {
		for _, file := range files {
			os.Remove(file)
		}
	}(tempFiles)

	// 获取当前时间并格式化
	if rpmRelease == "" {
		rpmRelease = defaultRpmRelease
	}
	cwd, _ := os.Getwd()

	log.Println("正在运行 Docker 容器构建 RPM...")
	dockerRunCmd := exec.Command("docker", "run", "--platform", "linux/amd64", "--rm", "-v", fmt.Sprintf("%s:/mac", cwd), "-e", fmt.Sprintf("RELEASE=%s", rpmRelease), dockerImage)
	dockerRunCmd.Stdout = os.Stdout
	dockerRunCmd.Stderr = os.Stderr
	if err := dockerRunCmd.Run(); err != nil {
		log.Fatalf("Docker run 失败: %v", err)
	}

	// 记录生成的 RPM 文件名
	rpmFileName := fmt.Sprintf("%s-%s-%s.el7.x86_64.rpm", appName, "1.0.0", rpmRelease)
	log.Printf("🍺 生成的 RPM 文件名：%s", rpmFileName)

	// 计算并输出耗时
	elapsedTime := time.Since(startTime)
	log.Printf("操作耗时: %s\n", elapsedTime)
}

func init() {
	buildRpmCmd.Flags().StringVarP(&appName, "app-name", "a", "", "应用名称")
	buildRpmCmd.Flags().StringVarP(&appSourcePath, "app-source-path", "s", "", "应用打包成品路径")
	buildRpmCmd.Flags().IntVarP(&tomcatPort, "tomcat-port", "t", 0, "TOMCAT 端口 (TOMCAT_PORT)")
	buildRpmCmd.Flags().StringVarP(&jvmSize, "jvm-size", "j", "medium", "JVM 大小 (JVM_SIZE)")
	buildRpmCmd.Flags().StringVarP(&cpuAffinity, "cpu-affinity", "c", "", "CPU 亲和性 (CPU_AFFINITY)")
	buildRpmCmd.Flags().StringVarP(&rpmRelease, "release", "r", defaultRpmRelease, "RPM Release Info")
	buildRpmCmd.Flags().StringVarP(&mainModule, "main-module", "m", "", "启动类")
	buildRpmCmd.Flags().BoolVarP(&debugImage, "debug", "d", false, "进入rpm构建容器内部")
}
