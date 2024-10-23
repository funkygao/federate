#!/bin/bash
#============================
# 私有化部署环境的启动JVM脚本
#============================

set -o errexit

BASEDIR=$(cd $(dirname $0) && pwd)/..     # 获取运行脚本的上级目录
readonly MAIN_MODULE="{{.MAIN_MODULE}}"   # main 函数所在的类名称。
JAVA="/export/servers/jdk1.8.0_191/bin/java"
APP_NAME="${APP_NAME:-wms6-reporting}"

CLASSPATH="$BASEDIR/conf/:$BASEDIR/lib/*"
JAVA_OPTS="$JAVA_OPTS -server -Xms${MEMORY_LIMIT} -Xmx${MEMORY_LIMIT} -XX:MaxMetaspaceSize=512m -XX:MetaspaceSize=512m -XX:MaxDirectMemorySize=1024m -XX:ConcGCThreads=1 -XX:ParallelGCThreads=4 -XX:CICompilerCount=2 -XX:+UseG1GC -XX:ErrorFile=${LOG_HOME}/java_error_%p.log -XX:+HeapDumpOnOutOfMemoryError -XX:HeapDumpPath=${LOG_HOME} "

# 获取当前应用的进程 id
function get_pid
{
   pgrep -f ".*DappName=$APP_NAME "
}

[[ -z $(get_pid) ]] || {
    echo "ERROR:  $APP_NAME already running" >&2
    exit 1
}

echo "Starting $APP_NAME ...."
[[ -x $JAVA ]] || {
    echo "ERROR: no executable java found at $JAVA" >&2
    exit 1
}

cd $BASEDIR
setsid "$JAVA" $JAVA_OPTS -Duser.country="CN" -Duser.language="zh" -Duser.timezone="Asia/Shanghai" -Dfile.encoding="UTF-8" -Ddubbo.hessian.allowNonSerializable=true -Dump.mode=off -DappName=$APP_NAME\
    -classpath "$CLASSPATH" \
    -Dbasedir="$BASEDIR" \
    $MAIN_MODULE \
    "$@"

sleep 0.5
[[ -n $(get_pid) ]] || {
    echo "ERROR: $APP_NAME failed to start" >&2
    exit 1
}
echo "$APP_NAME is up runnig :)"
