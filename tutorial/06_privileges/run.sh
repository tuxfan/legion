#sri


#for ele in 1024 16384 65536 131072 
for ele in 131072 
do
for itr in 8 128 512
do 
/usr/bin/time -f "time taken: %E Memory: %M" ./privileges -n $ele -i $itr 
#/usr/bin/time -f "time taken: %E Memory: %M" ./privileges -n $ele -i $itr -lg:resilient 
done
done
