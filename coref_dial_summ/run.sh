#if ! pgrep -x "gedit" > /dev/null
#then
#    echo "Stopped"
#fi
PID=29190
#if ! ps -p $PID > /dev/null
#then
#    python3 main.py
#fi
while [ -e /proc/$PID ]
do
    echo "Process: $PID is still running"
    sleep .6
done
python3 main.py