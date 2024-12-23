# Generated by federate, DO NOT EDIT.
SHELL := /bin/bash
.SILENT:

help:
	awk 'BEGIN {FS = ":.*##"; printf "Usage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } /^##[^@]/ { printf "%s\n", substr($$0, 4) }' $(MAKEFILE_LIST)

##@ Debug

clean:
	echo "☕️ mvn clean ..."
	mvn clean -q

check_package_dir:
	if [ ! -d "target/{{.AppName}}-1.0.0-SNAPSHOT-package" ]; then \
		echo "target/{{.AppName}}-1.0.0-SNAPSHOT-package not found. Execute 'make package' at project root dir"; \
		exit 1; \
	fi

run:check_package_dir ## Debug {{.AppName}} locally after configuring hosts.
	echo "☕️ Starting {{.AppName}} ..."
	env {{.Env}} java -DappName={{.AppName}} -DLOG_HOME=. -Xss1m -server -classpath target/{{.AppName}}-1.0.0-SNAPSHOT-package/conf:target/{{.AppName}}-1.0.0-SNAPSHOT-package/lib/* {{.ClassName}}

inspect:check_package_dir ## Inspect indirect conflict risk before runtime.
	java -DLOG_HOME=. -classpath target/{{.AppName}}-1.0.0-SNAPSHOT-package/conf:target/{{.AppName}}-1.0.0-SNAPSHOT-package/lib/* {{.ClassName}} --detect-indirect-risk

##@ Analysis

dependency-tree:check_package_dir ## Execute mvn dependency:tree.
	mvn dependency:tree -Dmaven.test.skip=true -DoutputFile=tree.txt -q
	echo "🍺 check out tree.txt"

{{- if ne .TomcatPort 0}}
##@ Deployment

build-rpm:check_package_dir ## Build {{.AppName}} into RPM to upload to yum repo.
	federate image build-rpm --app-name {{.AppName}} --app-source-path {{.AppSrc}} --jvm-size {{.JvmSize}} --main-module {{.ClassName}} --tomcat-port {{.TomcatPort}}

build-rpm-debug:
	federate image build-rpm --debug --app-name {{.AppName}}

build-docker: ## Build {{.AppName}} into Docker image to push to jdcloud image registry.
	federate image build-docker --image-repo {{.AppName}} --app-source-path {{.AppSrc}}
{{- end}}
