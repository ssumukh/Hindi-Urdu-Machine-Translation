######bash runGIZA++.sh src-file tgt-gile will create "src-tgt-gmap" file 
src=$1
trg=$2
plain2snt.out $src $trg
mkcls -n10 -p$src -V$trg.vcb.classes
mkcls -n10 -p$trg -V$src.vcb.classes
snt2cooc.out $src.vcb $trg.vcb $src"_"$trg".snt" > $src"_"$trg".cooc"
GIZA++ -ml 101 -S $src.vcb -T $trg.vcb -C $src"_"$trg".snt" -CoocurrenceFile $src"_"$trg".cooc"

mv *.A3.final $1-$2.gmap
rm *.cooc *.perp
rm *.final *.config *.gizacfg
rm *.snt *.vcb *.classes *.cats
