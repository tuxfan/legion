#sri

for i in 100 200 400 800 1000
do 
/usr/bin/time -f "time taken: %E Memory: %M" ./index_tasks 100 $i 
/usr/bin/time -f "time taken: %E Memory: %M" ./index_tasks 100 $i -lg:resilient 
done
