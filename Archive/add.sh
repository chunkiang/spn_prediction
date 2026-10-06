N=1
for name in 10mer_output/*txt
do

#new_name=$(echo $name | sed -n -e 's/10mer_output\/10mers-//g' -e 's/.txt//gp' )
#echo $new_name

awk '{print $1, $2, FILENAME}' $name | sed -n  -e 's/10mer_output\/10mers-//g' -e 's/.txt//gp' > ${name}_new
mv ${name}_new ${name}

echo ${name}   ${N}   $(date +"%Y-%m-%d %H:%M:%S")
N=$[${N}+1]

done
