.PHONY: build-rpm compile clean

all:compile

compile:clean
	@mvn package -U -Dmaven.test.skip=true -T8

clean:
	@mvn clean

build-rpm:
	@federate image build-rpm --app-name {{.AppName}} --app-source-path {{.AppSrc}} --jvm-size {{.JvmSize}} --main-module {{.ClassName}} --tomcat-port {{.TomcatPort}}
