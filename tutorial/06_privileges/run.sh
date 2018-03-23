#sri


for ele in 121072 
do
for itr in 128 256 512 1024
do 
/usr/bin/time -f "time taken: %E Memory: %M" ./privileges -n $ele -i $itr 
/usr/bin/time -f "time taken: %E Memory: %M" ./privileges -n $ele -i $itr -lg:resilient 
done
done
