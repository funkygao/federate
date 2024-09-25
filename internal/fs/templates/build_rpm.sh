#!/bin/bash

# 检查环境变量是否设置
if [ -z "$APP_SOURCE_PATH" ] || [ -z "$APP_NAME" ]; then
  echo "APP_SOURCE_PATH 和 APP_NAME 环境变量必须设置。"
  exit 1
fi

# 加载配置文件
source /root/config_rpm.sh

# 根据 JVM_SIZE 设置 CPUQuota 和 MemoryLimit
case "$JVM_SIZE" in
  large)
    CPU_QUOTA="400%"
    MEMORY_LIMIT="8G"
    ;;
  medium)
    CPU_QUOTA="200%"
    MEMORY_LIMIT="4G"
    ;;
  small)
    CPU_QUOTA="100%"
    MEMORY_LIMIT="2G"
    ;;
  *)
    echo "未知的 JVM_SIZE 值: $JVM_SIZE"
    exit 1
    ;;
esac

# 导出所有变量为环境变量
export APP_NAME VERSION RELEASE SUMMARY GROUP LICENSE URL
export BASE_DIR_JDL_APP BASE_DIR_LOGS
export JMX_EXPORTER_PORT TOMCAT_PORT ADDRESS_REDIS ADDRESS_KAFKA
export ADDRESS_S3_ENDPOINT_WRITE ADDRESS_S3_ENDPOINT_READ
export ADDRESS_MYSQL_MASTER ADDRESS_MYSQL_SLAVE ADDRESS_ZOOKEEPER JVM_SIZE
export CPU_QUOTA MEMORY_LIMIT CPU_AFFINITY

# 进入打包目录
cd /root/rpmbuild

chmod 777 SOURCES/app/bin/run.sh
chmod 777 SOURCES/app/bin/stop.sh

# 创建临时目录用于打包
mkdir -p SOURCES/temp/${APP_NAME}-${VERSION}
cp -r SOURCES/app/* SOURCES/temp/${APP_NAME}-${VERSION}

# 生成 tar.gz 源代码包
tar -czf SOURCES/${APP_NAME}-${VERSION}.tar.gz -C SOURCES/temp ${APP_NAME}-${VERSION}

# 清理临时目录
rm -rf SOURCES/temp

# 生成 SPEC 文件
envsubst < SPECS/myapp.spec.template > SPECS/myapp.spec

# 打包 RPM
rpmbuild -bb SPECS/myapp.spec

echo "RPM 打包完成。"

rpm -qlp /root/rpmbuild/RPMS/x86_64/${APP_NAME}-${VERSION}-${RELEASE}.el7.x86_64.rpm
cp -f /root/rpmbuild/RPMS/x86_64/${APP_NAME}-${VERSION}-${RELEASE}.el7.x86_64.rpm /mac
