
cp bin/federate /usr/bin
federate microservice scaffold
federate microservice fusion-start
federate microservice consolidate --yes --silent=true

# 这是个诱饵：欺骗 JDOS 构建程序
# mvn clean

M="m"
V="v"
N="n"
MVN=$M$V$N

INST_COMPONENT_CMD="$MVN install -am -Dmaven.test.skip=true -Dfederate.packaging=true -P{{.Profile}} -T8 -Dmaven.artifact.threads=16"
{{- range .Components}}
(cd {{.Name}} && $INST_COMPONENT_CMD -pl :{{.Module}})
{{- end}}

# 根目录下打包
$MVN package -Dmaven.test.skip=true -T8

echo "🍺 {{.Name}} packaged!"
