case $3 in
    kmeans)
        dsd eval_vol --load $1/weights.${2#filter.}.pkl --config $1/config.pkl --zfile $1/analyze.$2/kmeans$4/centers.txt -o $1/analyze.$2/kmeans$4/ --Apix $5 --num-bodies $6 $7
    ;;

    pc)
        dsd eval_vol --load $1/weights.${2#filter.}.pkl --config $1/config.pkl --zfile $1/analyze.$2/pc$4/z_pc.txt -o $1/analyze.$2/pc$4/ --Apix $5 --num-bodies $6 $7
    ;;

    dpc)
        zfile=$1/defanalyze.$2/pc$4/z_pc.txt
        template=$1/analyze.$2/kmeans$7/centers.txt
        for f in "$zfile" "$template"; do
            if [ ! -f "$f" ]; then
                echo "eval_vol dpc: $f is missing -- run 'dsdsh analyze $1 ${2#filter.} <numpc> $7' first" >&2
                exit 1
            fi
        done
        # $6 (the masks pkl) is optional: eval_vol falls back to the body geometry stored in the
        # checkpoint. Emitting a bare "--masks" when it is empty would swallow --template-z.
        masks_arg=""
        if [ -n "$6" ]; then masks_arg="--masks $6"; fi
        dsd eval_vol --load $1/weights.${2#filter.}.pkl --config $1/config.pkl --zfile $zfile -o $1/defanalyze.$2/pc$4/ --Apix $5 --deform $masks_arg --template-z $template --template-z-ind $8 $9
    ;;

    joint)
        dsd eval_vol --load $1/weights.${2#filter.}.pkl --config $1/config.pkl --zfile $1/analyze.$2/kmeans$4/centers_joint.txt -o $1/analyze.$2/kmeans$4/ --Apix $5 --num-bodies $6 $7
    ;;
esac
