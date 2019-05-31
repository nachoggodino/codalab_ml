declare -A optimal_lr
langs=( cr es mx pe uy )
#techniques=( bpe bert chars bpe:bert glove:bpe word2vec:bpe )
techniques=( bpe chars bert )

optimal_lr[bpe]=0.5
optimal_lr[bert]=0.05
optimal_lr[chars]=0.5

optimal_lr[bpe:bert]=0.1
optimal_lr[glove:bpe]=0.1
optimal_lr[word2vec:bpe]=0.2
optimal_lr[glove:word2vec:bpe]=0.2

for technique in "${techniques[@]}"; do 
	echo "===========  " $technique " ============="
	for lang in "${langs[@]}"; do
		echo "------------- Processing lang " $lang " ----------------"
		echo "***** MONO ******"
		# DEV
		python3 classifyFlair2.py -l $lang -e $technique -m ${optimal_lr[$technique]}
		# TST
		python3 classifyFlair2.py -t -l $lang -e $technique -m ${optimal_lr[$technique]}


		echo "***** CROSS ******"
		# DEV CROSS
		python3 classifyFlair2.py -c -l $lang -e $technique -m ${optimal_lr[$technique]}
		# TST CROSS
		python3 classifyFlair2.py -t -c  -l $lang -e $technique -m ${optimal_lr[$technique]}
	done
done