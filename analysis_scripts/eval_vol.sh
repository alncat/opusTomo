case $5 in
    kmeans)
        python -m cryodrgn.commands.eval_vol --load $1/weights.$2.pkl --config $1/config.pkl --zfile $1/analyze.$2/kmeans$3/centers.txt -o $1/analyze.$2/kmeans$3/ --Apix $4 --encode-mode grad --pe-type vanilla --template-type conv --num-bodies $6
    ;;

    pc)
        python -m cryodrgn.commands.eval_vol --load $1/weights.$2.pkl --config $1/config.pkl --zfile $1/analyze.$2/pc$3/z_pc.txt -o $1/analyze.$2/pc$3/ --Apix $4 --encode-mode grad --pe-type vanilla --template-type conv --num-bodies $6
    ;;

    dpc)
        python -m cryodrgn.commands.eval_vol --load $1/weights.$2.pkl --config $1/config.pkl --zfile $1/defanalyze.$2/pc$3/z_pc.txt -o $1/defanalyze.$2/pc$3/ --Apix $4 --encode-mode grad --pe-type vanilla --template-type conv --deform --masks $6 --template-z $1/analyze.$2/kmeans$7/centers.txt
    ;;
esac
