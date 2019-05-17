cat general-test-tagged-3l.ftx general-test-tagged.ftx general-test1k-tagged-3l.ftx general-test1k-tagged.ftx intertass-CR-development-tagged.ftx intertass-CR-train-tagged.ftx intertass-ES-development-tagged.ftx intertass-ES-train-tagged.ftx intertass-PE-development-tagged.ftx intertass-PE-train-tagged.ftx > a.ftx

fasttext supervised -input a.ftx -output pm -lr 0.3 -epoch 5 -wordNgrams 2 -ws 10
fasttext supervised -input intertass_pe_train.ftx -output class_pe_pre_3 -pretrainedVectors pm.vec -lr 0.3 -epoch 5 -wordNgrams 2 -ws 10
fasttext test class_pe_pre_3.bin intertass_pe_dev.ftx
fasttext predict-prob class_pe_pre_3.bin intertass_pe_dev.ftx 4 > dev_mono_out/pe_dev_mono.out
fasttext supervised -input intertass_pe_train_dev.ftx -output class_pe_tr_dev_3 -pretrainedVectors pm.vec -lr 0.3 -epoch 5 -wordNgrams 2 -ws 10
fasttext predict-prob class_pe_tr_dev_3.bin intertass_pe_test.ftx 4 > test_mono_out/pe_test_mono.out

less intertass_pe_test.xml | grep "<tweetid>" | sed 's/<tweetid>//g' | sed 's/<\/tweetid>//g' | awk '{print($1)}' > pe_test_tid.txt ; awk '{x=gsub("__label__","",$1);print $1}' < test_mono_out/pe_test_mono.out | paste pe_test_tid.txt -  > test_mono/pe.tsv
less intertass_pe_dev.xml | grep "<tweetid>" | sed 's/<tweetid>//g' | sed 's/<\/tweetid>//g' | awk '{print($1)}' > pe_dev_tid.txt ; awk '{x=gsub("__label__","",$1);print $1}' < dev_mono_out/pe_dev_mono.out | paste pe_dev_tid.txt -  > dev_mono/pe.tsv
python3 evaluate.py intertass_pe_dev_gold.tsv dev_mono/pe.tsv



fasttext supervised -input intertass_uy_train.ftx -output class_uy_pre_3 -pretrainedVectors pm.vec -lr 0.3 -epoch 5 -wordNgrams 2 -ws 10
fasttext test class_uy_pre_3.bin intertass_uy_dev.ftx
fasttext supervised -input intertass_uy_train_dev.ftx -output class_uy_tr_dev_3 -pretrainedVectors pm.vec -lr 0.3 -epoch 5 -wordNgrams 2 -ws 10


fasttext predict-prob class_uy_pre_3.bin intertass_uy_dev.ftx 4 > dev_mono_out/uy_dev_mono.out
less intertass_uy_dev.xml | grep "<tweetid>" | sed 's/<tweetid>//g' | sed 's/<\/tweetid>//g' | awk '{print($1)}' > uy_dev_tid.txt ; awk '{x=gsub("__label__","",$1);print $1}' < dev_mono_out/uy_dev_mono.out | paste uy_dev_tid.txt -  > dev_mono/uy.tsv

fasttext predict-prob class_uy_tr_dev_3.bin intertass_uy_test.ftx 4 > test_mono_out/uy_test_mono.out
less intertass_uy_dev.xml | grep "<tweetid>" | sed 's/<tweetid>//g' | sed 's/<\/tweetid>//g' | awk '{print($1)}' > uy_test_tid.txt ; awk '{x=gsub("__label__","",$1);print $1}' < test_mono_out/uy_test_mono.out | paste uy_test_tid.txt -  > test_mono/uy.tsv
python3 evaluate.py intertass_uy_dev_gold.tsv dev_mono/uy.tsv



fasttext supervised -input intertass_cr_train.ftx -output class_cr_pre_3 -pretrainedVectors pm.vec -lr 0.3 -epoch 5 -wordNgrams 2 -ws 10
fasttext test class_cr_pre_3.bin intertass_cr_dev.ftx
fasttext predict-prob class_cr_pre_3.bin intertass_cr_dev.ftx 4 > dev_mono_out/cr_dev_mono.out
fasttext supervised -input intertass_cr_train_dev.ftx -output class_cr_tr_dev_3 -pretrainedVectors pm.vec -lr 0.3 -epoch 5 -wordNgrams 2 -ws 10
fasttext predict-prob class_cr_tr_dev_3.bin intertass_cr_test.ftx 4 > test_mono_out/cr_test_mono.out

less intertass_cr_test.xml | grep "<tweetid>" | sed 's/<tweetid>//g' | sed 's/<\/tweetid>//g' | awk '{print($1)}' > cr_test_tid.txt ; awk '{x=gsub("__label__","",$1);print $1}' < test_mono_out/cr_test_mono.out | paste cr_test_tid.txt -  > test_mono/cr.tsv
less intertass_cr_dev.xml | grep "<tweetid>" | sed 's/<tweetid>//g' | sed 's/<\/tweetid>//g' | awk '{print($1)}' > cr_dev_tid.txt ; awk '{x=gsub("__label__","",$1);print $1}' < dev_mono_out/cr_dev_mono.out | paste cr_dev_tid.txt -  > dev_mono/cr.tsv
python3 evaluate.py intertass_cr_dev_gold.tsv dev_mono/cr.tsv