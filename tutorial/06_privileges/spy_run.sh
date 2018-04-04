./privileges -i $1 -logfile spy_%.log
$LG_RT_DIR/../tools/legion_spy.py -lpa spy_*.log
$LG_RT_DIR/../tools/legion_spy.py -dez spy_*.log
