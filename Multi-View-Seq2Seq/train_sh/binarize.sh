# Change the name of source file for pre-processing, as well as the destdir
fairseq-preprocess \
  --source-lang "source" \
  --target-lang "target" \
  --trainpref "../data/labels/train_sent_c99_label.bpe" \
  --validpref "../data/labels/val_sent_c99_label.bpe" \
  --destdir "new_dialogsum-bin/" \
  --workers 60 \
  --srcdict dict.txt \
  --tgtdict dict.txt;
  
# Change the name of source file for pre-processing, as well as the destdir
fairseq-preprocess \
  --source-lang "source" \
  --target-lang "target" \
  --trainpref "../data/labels/train_sent_trans_cons_label.bpe" \
  --validpref "../data/labels/val_sent_trans_cons_label.bpe" \
  --destdir "new_dialogsum-bin_2/" \
  --workers 60 \
  --srcdict dict.txt \
  --tgtdict dict.txt;
  
# Change the name of source file for pre-processing, as well as the destdir
fairseq-preprocess \
  --source-lang "source" \
  --target-lang "target" \
  --trainpref "../data/labels/train_sent_c99_label_p1.bpe" \
  --validpref "../data/labels/val_sent_c99_label_p1.bpe" \
  --destdir "new_dialogsum-bin/" \
  --workers 60 \
  --srcdict dict.txt \
  --tgtdict dict.txt;
  
# Change the name of source file for pre-processing, as well as the destdir
fairseq-preprocess \
  --source-lang "source" \
  --target-lang "target" \
  --trainpref "../data/labels/train_sent_trans_cons_label_p1.bpe" \
  --validpref "../data/labels/val_sent_trans_cons_label_p1.bpe" \
  --destdir "new_dialogsum-bin_2/" \
  --workers 60 \
  --srcdict dict.txt \
  --tgtdict dict.txt;

# Change the name of source file for pre-processing, as well as the destdir
fairseq-preprocess \
  --source-lang "source" \
  --target-lang "target" \
  --trainpref "../data/labels/train_sent_c99_label_p2.bpe" \
  --validpref "../data/labels/val_sent_c99_label_p2.bpe" \
  --destdir "new_dialogsum-bin/" \
  --workers 60 \
  --srcdict dict.txt \
  --tgtdict dict.txt;
  
# Change the name of source file for pre-processing, as well as the destdir
fairseq-preprocess \
  --target-lang "target" \
  --trainpref "../data/labels/train_sent_trans_cons_label_p2.bpe" \
  --validpref "../data/labels/val_sent_trans_cons_label_p2.bpe" \
  --destdir "new_dialogsum-bin_2/" \
  --workers 60 \
  --srcdict dict.txt \
  --tgtdict dict.txt;