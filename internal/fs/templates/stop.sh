PIDFILE="./start$1.pid"
echo $PIDFILE
SLEEP_INTERVAL=3
MAX_RETRY_TIMES=3
if [ ! -f "$PIDFILE" ]; then
  echo "no boot to stop (could not find file $PIDFILE)"
else
  COUNT=0
  while [ $COUNT -lt $MAX_RETRY_TIMES ]; do
    if kill -0 $(cat $PIDFILE) >/dev/null 2>&1; then
      echo killing process $(cat $PIDFILE).
      kill $(cat "$PIDFILE")
      echo "sleep $SLEEP_INTERVAL seconds ..."
      sleep $SLEEP_INTERVAL
      COUNT=$((COUNT + 1))
      echo try $COUNT times
    else
      echo process $(cat $PIDFILE) is killed
      break
    fi
  done

  if kill -0 $(cat $PIDFILE) >/dev/null 2>&1; then
    kill -9 $(cat $PIDFILE)
    echo execut kill -9 for $(cat $PIDFILE)
  fi

  rm -f "$PIDFILE"
  echo STOPPED
fi
exit 0
