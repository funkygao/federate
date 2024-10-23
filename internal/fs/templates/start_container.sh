#!/bin/bash

mirror_server='172.25.134.109'
docker_init_script='docker_initialize.sh'
uname -m |grep -q aarch64 && docker_init_script='docker_initialize_arm.sh'
grep -qi ubuntu /etc/issue && mkdir -p /run/sshd && docker_init_script='docker_initialize_ubuntu.sh'
docker_init_url="http://${mirror_server}/docker/${docker_init_script}"

while true;do
  if ping -c 1 $mirror_server > /dev/null 2>&1;then
    echo "network is normal"
    if [[ `curl -I -m 1 -o /dev/null -s -w %{http_code} $docker_init_url` == 200 ]];then
      echo 'http test success'
      break
    else
      echo 'http test error'
    fi  
  else
    echo "network is unnormal"
    ip ad sh
    ip ro sh
  fi  
  sleep 2
done

curl -sL  $docker_init_url | bash

######以上内容都是初始化系统依赖脚本，原样复制即可#######
######以下内容是启动应用内容，根据实际情况填写######
[ -f "/home/admin/start_before.sh" ] && su -m admin -c "bash /home/admin/start_before.sh"
# 比如代码在/opt目录下，运行依赖/export/log/redis，则需创建并给相关目录赋权
mkdir -p /export/log/redis/
chown -R admin:admin /opt/ /export/log/redis/
# 必须要切换到admin用户去启动应用
su -m admin -c "bash /home/admin/start.sh"
