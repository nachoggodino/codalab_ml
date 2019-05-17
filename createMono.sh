# CR
python3 createText4FastText.py intertass_cr_train.xml
python3 createText4FastText.py intertass_cr_dev.xml
python3 createText4FastText_nolbl.py intertass_cr_test.xml

cat intertass_cr_train.ftx intertass_cr_dev.ftx > intertass_cr_train_dev.ftx

fasttext supervised -input intertass_cr_train.ftx -output class_cr_preT1 -pretrainedVectors pretrained_model.vec -lr 1.0 -epoch 5 -wordNgrams 2
fasttext test class_cr_preT1.bin intertass_cr_dev.ftx
fasttext predict-prob class_cr_preT1.bin intertass_cr_dev.ftx 4 > dev_mono_out/cr_dev_mono.out

fasttext supervised -input intertass_cr_train_dev.ftx -output class_cr_tr_dev -pretrainedVectors pretrained_model.vec -lr 1.0 -epoch 5 -wordNgrams 2
fasttext predict-prob class_cr_tr_dev.bin intertass_cr_test.ftx 4 > test_mono_out/cr_test_mono.out

less intertass_cr_dev.xml | grep "<tweetid>" | sed 's/<tweetid>//g' | sed 's/<\/tweetid>//g' | awk '{print($1)}' > cr_dev_tid.txt ; awk '{x=gsub("__label__","",$1);print $1}' < dev_mono_out/cr_dev_mono.out | paste cr_dev_tid.txt -  > dev_mono/cr.tsv
less intertass_cr_test.xml | grep "<tweetid>" | sed 's/<tweetid>//g' | sed 's/<\/tweetid>//g' | awk '{print($1)}' > cr_test_tid.txt ; awk '{x=gsub("__label__","",$1);print $1}' < test_mono_out/cr_test_mono.out | paste cr_test_tid.txt -  > test_mono/cr.tsv

python3 evaluate.py intertass_cr_dev_gold.tsv dev_mono/cr.tsv



# ES
python3 createText4FastText.py intertass_es_train.xml
python3 createText4FastText.py intertass_es_dev.xml
python3 createText4FastText_nolbl.py intertass_es_test.xml

cat intertass_es_train.ftx intertass_es_dev.ftx > intertass_es_train_dev.ftx

fasttext supervised -input intertass_es_train.ftx -output class_es_preT1 -pretrainedVectors pretrained_model.vec -lr 1.0 -epoch 3 -wordNgrams 2
fasttext test class_es_preT1.bin intertass_es_dev.ftx
fasttext predict-prob class_es_preT1.bin intertass_es_dev.ftx 4 > dev_mono_out/es_dev_mono.out

fasttext supervised -input intertass_es_train_dev.ftx -output class_es_tr_dev -pretrainedVectors pretrained_model.vec -lr 1.0 -epoch 3 -wordNgrams 2
fasttext predict-prob class_es_tr_dev.bin intertass_es_test.ftx 4 > test_mono_out/es_test_mono.out
less intertass_es_dev.xml | grep "<tweetid>" | sed 's/<tweetid>//g' | sed 's/<\/tweetid>//g' | awk '{print($1)}' > es_dev_tid.txt ; awk '{x=gsub("__label__","",$1);print $1}' < dev_mono_out/es_dev_mono.out | paste es_dev_tid.txt -  > dev_mono/es.tsv
less intertass_es_test.xml | grep "<tweetid>" | sed 's/<tweetid>//g' | sed 's/<\/tweetid>//g' | awk '{print($1)}' > es_test_tid.txt ; awk '{x=gsub("__label__","",$1);print $1}' < test_mono_out/es_test_mono.out | paste es_test_tid.txt -  > test_mono/es.tsv

python3 evaluate.py intertass_es_dev_gold.tsv dev_mono/es.tsv

# MX
python3 createText4FastText.py intertass_mx_train.xml
python3 createText4FastText.py intertass_mx_dev.xml
python3 createText4FastText_nolbl.py intertass_mx_test.xml

cat intertass_mx_train.ftx intertass_mx_dev.ftx > intertass_mx_train_dev.ftx

fasttext supervised -input intertass_mx_train.ftx -output class_mx_preT1 -pretrainedVectors pretrained_model.vec -lr 0.5 -epoch 5 -wordNgrams 2
fasttext test class_mx_preT1.bin intertass_mx_dev.ftx
fasttext predict-prob class_mx_preT1.bin intertass_mx_dev.ftx 4 > dev_mono_out/mx_dev_mono.out

fasttext supervised -input intertass_mx_train_dev.ftx -output class_mx_tr_dev -pretrainedVectors pretrained_model.vec -lr 0.5 -epoch 5 -wordNgrams 2
fasttext predict-prob class_mx_tr_dev.bin intertass_mx_test.ftx 4 > test_mono_out/mx_test_mono.out
less intertass_mx_dev.xml | grep "<tweetid>" | sed 's/<tweetid>//g' | sed 's/<\/tweetid>//g' | awk '{print($1)}' > mx_dev_tid.txt ; awk '{x=gsub("__label__","",$1);print $1}' < dev_mono_out/mx_dev_mono.out | paste mx_dev_tid.txt -  > dev_mono/mx.tsv
less intertass_mx_test.xml | grep "<tweetid>" | sed 's/<tweetid>//g' | sed 's/<\/tweetid>//g' | awk '{print($1)}' > mx_test_tid.txt ; awk '{x=gsub("__label__","",$1);print $1}' < test_mono_out/mx_test_mono.out | paste mx_test_tid.txt -  > test_mono/mx.tsv

python3 evaluate.py intertass_mx_dev_gold.tsv dev_mono/mx.tsv

# PE
python3 createText4FastText.py intertass_pe_train.xml
python3 createText4FastText.py intertass_pe_dev.xml
python3 createText4FastText_nolbl.py intertass_pe_test.xml

cat intertass_pe_train.ftx intertass_pe_dev.ftx > intertass_pe_train_dev.ftx

fasttext supervised -input intertass_pe_train.ftx -output class_pe_preT1 -pretrainedVectors pretrained_model.vec -lr 1.0 -epoch 8 -wordNgrams 2
fasttext test class_pe_preT1.bin intertass_pe_dev.ftx
fasttext predict-prob class_pe_preT1.bin intertass_pe_dev.ftx 4 > dev_mono_out/pe_dev_mono.out

fasttext supervised -input intertass_pe_train_dev.ftx -output class_pe_tr_dev -pretrainedVectors pretrained_model.vec -lr 1.0 -epoch 10 -wordNgrams 2
fasttext predict-prob class_pe_tr_dev.bin intertass_pe_test.ftx 4 > test_mono_out/pe_test_mono.out

less intertass_pe_dev.xml | grep "<tweetid>" | sed 's/<tweetid>//g' | sed 's/<\/tweetid>//g' | awk '{print($1)}' > pe_dev_tid.txt ; awk '{x=gsub("__label__","",$1);print $1}' < dev_mono_out/pe_dev_mono.out | paste pe_dev_tid.txt -  > dev_mono/pe.tsv
less intertass_pe_test.xml | grep "<tweetid>" | sed 's/<tweetid>//g' | sed 's/<\/tweetid>//g' | awk '{print($1)}' > pe_test_tid.txt ; awk '{x=gsub("__label__","",$1);print $1}' < test_mono_out/pe_test_mono.out | paste pe_test_tid.txt -  > test_mono/pe.tsv

python3 evaluate.py intertass_pe_dev_gold.tsv dev_mono/pe.tsv


# UY
python3 createText4FastText.py intertass_uy_train.xml
python3 createText4FastText.py intertass_uy_dev.xml
python3 createText4FastText_nolbl.py intertass_uy_test.xml

cat intertass_uy_train.ftx intertass_uy_dev.ftx > intertass_uy_train_dev.ftx

fasttext supervised -input intertass_uy_train.ftx -output class_uy_preT1 -pretrainedVectors pretrained_model.vec -lr 1.0 -epoch 10 -wordNgrams 2
fasttext test class_uy_preT1.bin intertass_uy_dev.ftx
fasttext predict-prob class_uy_preT1.bin intertass_uy_dev.ftx 4 > dev_mono_out/uy_dev_mono.out

fasttext supervised -input intertass_uy_train_dev.ftx -output class_uy_tr_dev -pretrainedVectors pretrained_model.vec -lr 1.0 -epoch 10 -wordNgrams 2
fasttext predict-prob class_uy_tr_dev.bin intertass_uy_test.ftx 4 > test_mono_out/uy_test_mono.out

less intertass_uy_dev.xml | grep "<tweetid>" | sed 's/<tweetid>//g' | sed 's/<\/tweetid>//g' | awk '{print($1)}' > uy_dev_tid.txt ; awk '{x=gsub("__label__","",$1);print $1}' < dev_mono_out/uy_dev_mono.out | paste uy_dev_tid.txt -  > dev_mono/uy.tsv
less intertass_uy_test.xml | grep "<tweetid>" | sed 's/<tweetid>//g' | sed 's/<\/tweetid>//g' | awk '{print($1)}' > uy_test_tid.txt ; awk '{x=gsub("__label__","",$1);print $1}' < test_mono_out/uy_test_mono.out | paste uy_test_tid.txt -  > test_mono/uy.tsv

python3 evaluate.py intertass_uy_dev_gold.tsv dev_mono/uy.tsv
