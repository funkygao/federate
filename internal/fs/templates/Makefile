.SILENT:
.PHONY: build-rpm package clean

clean:
	@mvn clean

package:clean
	@mvn package -U -Dmaven.test.skip=true -T8

build-rpm:package
	@federate image build-rpm --app-name {{.AppName}} --app-source-path {{.AppSrc}} --jvm-size {{.JvmSize}} --main-module {{.ClassName}} --tomcat-port {{.TomcatPort}}

build-rpm-debug:
	@federate image build-rpm --debug --app-name {{.AppName}}
