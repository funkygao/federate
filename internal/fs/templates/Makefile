.SILENT:
.PHONY: build-rpm build-docker package run clean help

##@ General

help: ## Display this help.
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Build

clean:
	@mvn clean

package:clean ## Compile and package {{.AppName}}.
	@mvn package -U -Dmaven.test.skip=true -T8

run: ## Bootstrap {{.AppName}} foreground on your host.
	@if [ ! -d "target/{{.AppName}}-1.0.0-SNAPSHOT-package" ]; then \
		echo "target/{{.AppName}}-1.0.0-SNAPSHOT-package not found. Running 'make package'..."; \
		$(MAKE) package; \
	fi
	@env {{.Env}} java -DappName={{.AppName}} -Xss1m -server -classpath target/{{.AppName}}-1.0.0-SNAPSHOT-package/conf:target/{{.AppName}}-1.0.0-SNAPSHOT-package/lib/* {{.ClassName}}

##@ Deployment

build-rpm:package ## Build {{.AppName}} into RPM to upload to yum repo.
	@federate image build-rpm --app-name {{.AppName}} --app-source-path {{.AppSrc}} --jvm-size {{.JvmSize}} --main-module {{.ClassName}} --tomcat-port {{.TomcatPort}}

build-rpm-debug:
	@federate image build-rpm --debug --app-name {{.AppName}}

build-docker:package ## Build {{.AppName}} into Docker image to push to jdcloud image registry.
	@federate image build-docker --image-repo {{.AppName}} --app-source-path {{.AppSrc}}
