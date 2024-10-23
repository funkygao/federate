#=============================
# 根据资源分配表格填写如下内容
#=============================
TOMCAT_PORT={{.TOMCAT_PORT}}
JMX_EXPORTER_PORT={{.JMX_EXPORTER_PORT}}

# 你的run.sh Xms/Xmx 参数如果引用了 MEMORY_LIMIT 则自动计算，否则需要手动配置
# 如果配置与 JVM_SIZE 不一致，可能 OOM killed：large: 8G, medium: 4G, small: 2G
JVM_SIZE="{{.JVM_SIZE}}" 

# 在运维工具上自动计算 CPU亲和性，例如，CPU_AFFINITY="4 5 6 7"
# CPU亲和性需要所有应用要么全都配置，要么全不配置
CPU_AFFINITY="{{.CPU_AFFINITY}}"

#=============================
# 中间件地址：核对端口号
#=============================
ADDRESS_REDIS="master.redis.local:6379"
ADDRESS_KAFKA="bootstrap.kafka.local:9092"
ADDRESS_S3_ENDPOINT_WRITE="endpoint.s3.local:9000"
ADDRESS_S3_ENDPOINT_READ="endpoint.s3.local:9000"
ADDRESS_MYSQL_MASTER="master.mysql.local:3358"
ADDRESS_MYSQL_SLAVE="slave.mysql.local:3358"
ADDRESS_ZOOKEEPER="n1.zk.local:2181,n2.zk.local:2181,n3.zk.local:2181"

#=============================
# base dir
#=============================
BASE_DIR_JDL_APP=/home/jdl
BASE_DIR_LOGS=/home/Logs

#=============================
# 以下全部内容无需修改
#=============================
VERSION=1.0.0
RELEASE=${RELEASE}
SUMMARY="JDL Application"
GROUP="Applications/System"
LICENSE="GPL"
URL="http://jdl.com"
