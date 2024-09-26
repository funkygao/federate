SHDIR=$(cd $(dirname $0) ; pwd)
echo current path:$SHDIR

PIDFILE="./start.pid"
if [ -f $PIDFILE ]; then
    if kill -0 `cat $PIDFILE` > /dev/null 2>&1; then
        echo server already running as process `cat $PIDFILE`. 
        exit 0
    fi
fi

if [ -z $START_JAVA_OPTS ]; then
    if [ $JDOS_MEMORY ]; then
        MemUse=$((${JDOS_MEMORY%%G}*1024/4))
        START_JAVA_OPTS="-Xmx"${MemUse}"m -Xms"${MemUse}"m"
    else
        START_JAVA_OPTS="-Xms2048m -Xmx2048m -XX:MaxPermSize=256m"
    fi
fi

# exec 
nohup java -DappName={{.AppName}} $START_JAVA_OPTS -Xss1m -server -classpath $SHDIR/../conf/:$SHDIR/../lib/* {{.MainClass}} &

# wirte pid to file
if [ $? -eq 0 ] 
then
    if /bin/echo -n $! > "$PIDFILE"
    then
        sleep 1
        echo STARTED SUCCESS
    else
        echo FAILED TO WRITE PID
        exit 1
    fi
# tail -100f $LOGFILE
else
    echo SERVER DID NOT START
    exit 1
fi
