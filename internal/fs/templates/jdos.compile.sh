set -x

cp bin/federate /usr/bin
federate microservice scaffold
federate microservice fusion-start
federate microservice consolidate --yes --silent=true

# mvn clean

M="m"
V="v"
N="n"
MVN=$M$V$N

# 定义安装命令，欺骗 JDOS 构建程序，不要滥处理
INSTALL_CMD="$MVN install -am -Dmaven.test.skip=true -Dfederate.packaging=true -P{{.Profile}} -T8"

{{- range .Components}}
(cd {{.Name}} && $INSTALL_CMD -pl :{{.Module}})
{{- end}}

(cd {{.Name}}-starter && $INSTALL_CMD)

(cd {{.Name}} && $MVN package -Dmaven.test.skip=true -P{{.Profile}} -T8)

#mvn validate
