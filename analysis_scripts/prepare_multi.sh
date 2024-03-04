python -m cryodrgn.commands.parse_multi_pose_star $1/$2.star -D $3 --Apix $4 -o $1/$2_pose_euler.pkl --masks $5.star $6 --bodies $7 --volumes $8
python -m cryodrgn.commands.parse_ctf_star $1/$2.star -D $3 --Apix $4 -o $1/$2_ctf.pkl $6
