MODELO EMBEDDINGS
fasttext supervised -input a.ftx -output pm -lr 0.3 -epoch 5 -wordNgrams 2 -ws 10

# CR
cat intertass_es_train.ftx intertass_es_dev.ftx intertass_mx_train.ftx intertass_mx_dev.ftx intertass_pe_train.ftx intertass_pe_dev.ftx intertass_uy_train.ftx intertass_uy_dev.ftx > intertass_cr_train_dev_cross.ftx

fasttext supervised -input intertass_cr_train_dev_cross.ftx -output class_cr_cross -pretrainedVectors pm.vec -lr 0.3 -epoch 5 -wordNgrams 2 -ws 10
fasttext test class_cr_cross.bin intertass_cr_dev.ftx
fasttext predict-prob class_cr_cross.bin intertass_cr_dev.ftx 4 > dev_cross_out/cr_dev_cross.out
fasttext predict-prob class_cr_cross.bin intertass_cr_test.ftx 4 > test_cross_out/cr_test_cross.out

less intertass_cr_dev.xml | grep "<tweetid>" | sed 's/<tweetid>//g' | sed 's/<\/tweetid>//g' | awk '{print($1)}' > cr_dev_tid.txt ; awk '{x=gsub("__label__","",$1);print $1}' < dev_cross_out/cr_dev_cross.out | paste cr_dev_tid.txt -  > dev_cross/cr.tsv
less intertass_cr_test.xml | grep "<tweetid>" | sed 's/<tweetid>//g' | sed 's/<\/tweetid>//g' | awk '{print($1)}' > cr_test_tid.txt ; awk '{x=gsub("__label__","",$1);print $1}' < test_cross_out/cr_test_cross.out | paste cr_test_tid.txt -  > test_cross/cr.tsv

python3 evaluate.py intertass_cr_dev_gold.tsv dev_cross/cr.tsv

# ES
cat intertass_cr_train.ftx intertass_cr_dev.ftx intertass_mx_train.ftx intertass_mx_dev.ftx intertass_pe_train.ftx intertass_pe_dev.ftx intertass_uy_train.ftx intertass_uy_dev.ftx > intertass_es_train_dev_cross.ftx

fasttext supervised -input intertass_es_train_dev_cross.ftx -output class_es_cross -pretrainedVectors pretrained_model.vec -lr 1.0 -epoch 5 -wordNgrams 2
fasttext test class_es_cross.bin intertass_es_dev.ftx
fasttext predict-prob class_es_cross.bin intertass_es_dev.ftx 4 > dev_cross_out/es_dev_cross.out
fasttext predict-prob class_es_cross.bin intertass_es_test.ftx 4 > test_cross_out/es_test_cross.out

less intertass_es_dev.xml | grep "<tweetid>" | sed 's/<tweetid>//g' | sed 's/<\/tweetid>//g' | awk '{print($1)}' > es_dev_tid.txt ; awk '{x=gsub("__label__","",$1);print $1}' < dev_cross_out/es_dev_cross.out | paste es_dev_tid.txt -  > dev_cross/es.tsv
less intertass_es_test.xml | grep "<tweetid>" | sed 's/<tweetid>//g' | sed 's/<\/tweetid>//g' | awk '{print($1)}' > es_test_tid.txt ; awk '{x=gsub("__label__","",$1);print $1}' < test_cross_out/es_test_cross.out | paste es_test_tid.txt -  > test_cross/es.tsv

python3 evaluate.py intertass_es_dev_gold.tsv dev_cross/es.tsv

# MX
cat intertass_cr_train.ftx intertass_cr_dev.ftx intertass_es_train.ftx intertass_es_dev.ftx intertass_pe_train.ftx intertass_pe_dev.ftx intertass_uy_train.ftx intertass_uy_dev.ftx > intertass_mx_train_dev_cross.ftx

fasttext supervised -input intertass_mx_train_dev_cross.ftx -output class_mx_cross -pretrainedVectors pretrained_model.vec -lr 1.0 -epoch 10 -wordNgrams 2
fasttext test class_mx_cross.bin intertass_mx_dev.ftx
fasttext predict-prob class_mx_cross.bin intertass_mx_dev.ftx 4 > dev_cross_out/mx_dev_cross.out
fasttext predict-prob class_mx_cross.bin intertass_mx_test.ftx 4 > test_cross_out/mx_test_cross.out

less intertass_mx_dev.xml | grep "<tweetid>" | sed 's/<tweetid>//g' | sed 's/<\/tweetid>//g' | awk '{print($1)}' > mx_dev_tid.txt ; awk '{x=gsub("__label__","",$1);print $1}' < dev_cross_out/mx_dev_cross.out | paste mx_dev_tid.txt -  > dev_cross/mx.tsv
less intertass_mx_test.xml | grep "<tweetid>" | sed 's/<tweetid>//g' | sed 's/<\/tweetid>//g' | awk '{print($1)}' > mx_test_tid.txt ; awk '{x=gsub("__label__","",$1);print $1}' < test_cross_out/mx_test_cross.out | paste mx_test_tid.txt -  > test_cross/mx.tsv

python3 evaluate.py intertass_mx_dev_gold.tsv dev_cross/mx.tsv

# PE
cat intertass_cr_train.ftx intertass_cr_dev.ftx intertass_es_train.ftx intertass_es_dev.ftx intertass_mx_train.ftx intertass_mx_dev.ftx intertass_uy_train.ftx intertass_uy_dev.ftx > intertass_pe_train_dev_cross.ftx

fasttext supervised -input intertass_pe_train_dev_cross.ftx -output class_pe_cross -pretrainedVectors pm.vec -lr 0.3 -epoch 5 -wordNgrams 2 -ws 10
fasttext test class_pe_cross.bin intertass_pe_dev.ftx
fasttext predict-prob class_pe_cross.bin intertass_pe_dev.ftx 4 > dev_cross_out/pe_dev_cross.out
fasttext predict-prob class_pe_cross.bin intertass_pe_test.ftx 4 > test_cross_out/pe_test_cross.out

less intertass_pe_dev.xml | grep "<tweetid>" | sed 's/<tweetid>//g' | sed 's/<\/tweetid>//g' | awk '{print($1)}' > pe_dev_tid.txt ; awk '{x=gsub("__label__","",$1);print $1}' < dev_cross_out/pe_dev_cross.out | paste pe_dev_tid.txt -  > dev_cross/pe.tsv
less intertass_pe_test.xml | grep "<tweetid>" | sed 's/<tweetid>//g' | sed 's/<\/tweetid>//g' | awk '{print($1)}' > pe_test_tid.txt ; awk '{x=gsub("__label__","",$1);print $1}' < test_cross_out/pe_test_cross.out | paste pe_test_tid.txt -  > test_cross/pe.tsv

python3 evaluate.py intertass_pe_dev_gold.tsv dev_cross/pe.tsv

# UY
cat intertass_cr_train.ftx intertass_cr_dev.ftx intertass_es_train.ftx intertass_es_dev.ftx intertass_mx_train.ftx intertass_mx_dev.ftx intertass_pe_train.ftx intertass_pe_dev.ftx  > intertass_uy_train_dev_cross.ftx

fasttext supervised -input intertass_uy_train_dev_cross.ftx -output class_uy_cross -pretrainedVectors pm.vec -lr 0.3 -epoch 5 -wordNgrams 2 -ws 10
fasttext test class_uy_cross.bin intertass_uy_dev.ftx
fasttext predict-prob class_uy_cross.bin intertass_uy_dev.ftx 4 > dev_cross_out/uy_dev_cross.out
fasttext predict-prob class_uy_cross.bin intertass_pe_test.ftx 4 > test_cross_out/uy_test_cross.out

less intertass_uy_dev.xml | grep "<tweetid>" | sed 's/<tweetid>//g' | sed 's/<\/tweetid>//g' | awk '{print($1)}' > uy_dev_tid.txt ; awk '{x=gsub("__label__","",$1);print $1}' < dev_cross_out/uy_dev_cross.out | paste uy_dev_tid.txt -  > dev_cross/uy.tsv
less intertass_uy_test.xml | grep "<tweetid>" | sed 's/<tweetid>//g' | sed 's/<\/tweetid>//g' | awk '{print($1)}' > uy_test_tid.txt ; awk '{x=gsub("__label__","",$1);print $1}' < test_cross_out/uy_test_cross.out | paste uy_test_tid.txt -  > test_cross/uy.tsv

python3 evaluate.py intertass_uy_dev_gold.tsv dev_cross/uy.tsv