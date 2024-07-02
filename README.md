# Table of contents
2. [Opus-TOMO](#opustomo)
    1. [80S ribosome](#80S)
    2. [FAS](#fas)
4. [setup environment](#setup)
5. [prepare data](#preparation)
6. [training](#training)
   1. [train_tomo](#train_tomo)
8. [analyze result](#analysis)
   1. [sample latent spaces](#sample)
   2. [reconstruct volumes](#reconstruct)
   3. [select particles](#select)

# Opus-TOMO <div id="opustomo">
This repository contains the implementation of opus-tomography (OPUS-TOMO), which is developed by Zhenwei (Benedict，本笃) Luo at the group of
Prof. Jianpeng Ma at Fudan University. The preprint of OPUS-TOMO is available at https://drive.google.com/drive/folders/1FcF1PC-0lY2C6DP0zP7K7qhcz_ltn5t-?usp=sharing.  Exemplar movies of the OPUS-TOMO is shown below:
A part of translation elongation cycle resolved by traversing PC8, which shows the translocation of A/T- and P- site tRNAs to A/P- and P-E site tRNAs.

https://github.com/alncat/opusTomo/assets/3967300/4fb1a46c-97bf-45ba-868d-b108cc2862f8

The binding and disassociation of different cofactors resolved by traversing PC8

https://github.com/alncat/opusTomo/assets/3967300/96d3bd65-26d0-4299-81a2-f1a72d88b1c2


The movement of L1 stalk resolved by traversing PC12

https://github.com/alncat/opusTomo/assets/3967300/d324098d-3cf8-407d-8091-ab738b7ae343


The main functionality of OPUS-TOMO is reconstructing conformational and compositonal changes from cryo-ET data end-to-end!
OPUS-TOMO can not only disentangle 3D structural information by reconstructing different conformations, but also reconstruct continous conformational dynamics for the macromolecules in cellular environment. The capacity of OPUS-TOMO is **robust against subtomograms from all kinds of particle localization methods**, such as neural-network based DeePiCt (https://github.com/ZauggGroup/DeePiCt), and the most crude templated matching in PyTom (https://github.com/SBC-Utrecht/PyTom/). OPUS-TOMO also enables the structural heterogeneity anlaysis using **the simplest statistic methods, PCA and KMeans clustering**. These two approaches can lead to sufficiently rich discovery about the conformational and compositional changes of macromolecules.

This project seeks to unravel how a latent space, encoding 3D structural information, can be learned by utilizing subtomograms which are aligned against a consensus reference model by subtomogram averaging.

An informative latent space is pivotal as it simplifies data analysis by providing a structured and reduced-dimensional representation of the data.

Our approach strategically **leverages the inevitable pose assignment errors introduced during consensus refinement, while concurrently mitigating their impact on the quality of 3D reconstructions**. Although it might seem paradoxical to exploit errors while minimizing their effects, our method has proven effective in achieving this delicate balance.

The workflow of OPUS-TOMO is demonstrated as follows:
<img width="763" alt="image" src="https://github.com/alncat/opusTomo/assets/3967300/ac975578-97c9-45cb-b909-e3fc275ff815">


Note that all input and output of this method are in real space! 
The architecture of encoder is (Encoder class in cryodrgn/models.py):
<img width="999" alt="image" src="https://github.com/alncat/opusTomo/assets/3967300/1d09533d-7265-4913-b140-d9c9c2e5b25b">



The architecture of decoder is (ConvTemplate class in cryodrgn/models.py. In this version, the default size of output volume is set to 192^3, I downsampled the intermediate activations to save some gpu memories. You can tune it as you wish, happy training!):

![Alt text](https://raw.githubusercontent.com/alncat/opusDSD/main/example/decoder.png?raw=true "Opus-DSD decoder")

The architecture of pose corrector is:

<img width="409" alt="image" src="https://github.com/alncat/opusTomo/assets/3967300/7f0eac9a-ecc2-47b8-ad43-81674b62c326">



## S.pombe 80S Ribosome <a name="80s"></a>
The S.pombe dataset is publicly available at EMPIAR-10988 (https://www.ebi.ac.uk/empiar/EMPIAR-10988/).
In this dataset, OPUS-TOMO has shown superior structural disentanglement ability to capture continous structral changes into PCs of the composition latent space, and characterizes 
structurally heterogeneous subpopulations.

<img width="781" alt="image" src="https://github.com/alncat/opusTomo/assets/3967300/b2211c66-28f3-4562-a1f5-de77444aff04">



## FAS <a name="fas"></a>

It can even reconstruct higher resolution structure for FAS in EMPIAR-10988 by clustering 221 particles from 4800 noisy subtomograms picked by template matching!

<img width="1107" alt="image" src="https://github.com/alncat/opusTomo/assets/3967300/3ae62232-b3d3-4c00-91a1-1b1a1714a59d">

Class 19 as a distinct cluster in UMAP projection shows the densities below:
<img width="740" alt="image" src="https://github.com/alncat/opusTomo/assets/3967300/9ad2da65-36cb-474a-be65-943b398f0294">



# set up environment <a name="setup"></a>

After cloning the repository, to run this program, you need to have an environment with pytorch and a machine with GPUs. The recommended configuration is a machine with 4 V100 GPUs.
You can create the conda environment for OPUS-TOMO using one of the environment files in the folder by executing

```
conda env create --name opustomo -f environmentcu11torch11.yml
```

This environment primarily contains cuda 11.3 and pytorch 1.11.0. To create an environment with cuda 11.3 and pytorch 1.10.1, you can choose ```environmentcu11.yml```. Lastly, ```environment.yml``` contains cuda 10.2 and pytorch 1.11.0. On V100 GPU, OPUS-TOMO with cuda 11.3 is 20% faster than OPUS-TOMO with cuda 10.2. However, it's worth noting that OPUS-TOMO **has not been tested on Pytorch version higher than 1.11.0**. We recommend using pytorch version 1.10.1 or 1.11.0. After the environment is sucessfully created, you can then activate it and execute our program within this environement.

```
conda activate opustomo
```

You can then install OPUS-TOMO by changing to the directory with cloned repository, and execute
```
pip install -e .
```

OPUS-TOMO can be kept up to date by 
```
git pull
```

The inference pipeline of our program can run on any GPU which supports cuda 10.2 or 11.3 and is fast to generate a 3D volume. However, the training of our program takes larger amount memory, we recommend using V100 GPUs at least.

# prepare data <a name="preparation"></a>

**Data Preparation Guidelines:**
1. **Cryo-ET Dataset:** OPUS-TOMO takes inputs for performing subtomogram averaging in Relion 3.0.8, which consists of subtomograms and the CTF parameters of tilts for each subtomogram in a starfile. Ensure that the cryo-ET dataset is stored as separate subtomograms in directory. A good dataset for tutorial is the S.pombe which is available at https://empiar.pdbj.org/entry/10180/ (It contains the coordinates for subtomograms and tilt alignment parameters for reconstructing tomograms.) We also have a script adapted from Relion for preparing the subtomograms and ctf starfiles. It is named as relion_ctf_prepare.py in the root folder of this repository. You can follow the detailed data preparation process in this paper: https://www.nature.com/articles/nprot.2016.124 .

2. **Subtomogram averaging Result:** The program requires a subtomogram averaging result, which should not apply any symmetry and must be stored as a Relion STAR file.

   In more details, the data preparation process commonly consists of **subtomogram extraction, ctf reconstruction, and subtomogram averaging**.
Commonly, the data should organized as: using separate directories to store each tomogram and its corresponding micrographs, the coordinate files for subtomograms from a tomogram are put in the directory of that tomogram, and the CTF estimation results are put in the same directory as the tomogram, which looks like:
```
ls -1 TS_026_Imod/
ctfplotter.com
ctfplotter.defocus
newst.com
tilt.com
TS_026_st.ali
TS_026_st.aln
TS_026_st.coords
TS_026_st.mrc
TS_026_st.mrcs
TS_026_st.order
TS_026_st.tlt
TS_026_st.xf
TS_026_st.xtilt
```

In the subtomogram extraction phase, you should have the picked coordinates, and tomograms. Put the coordinate files into the same directory as the tomogram. The subtomograms can be extracted by RELION 3.0.8 using the command
```
relion_preprocess --coord_suffix .coords --coord_dir ./ --part_dir Extract/extract_tomo/ --extract --bg_radius 44 --norm --extract_size 128 --i all_tomograms.star
```
where coord_suffix specifies the suffix of coordinate files, coord_idr ./ tells relion_preprocess that the coords files are in the same folder as the tomogram, part_dir specifies the output directory for subtomograms, bg_radius specifies the radius for circle mask, extract_size specifies the size of subtomogram, and i specifies the starfile with the path of tomograms, which has the following contents:
```
data_
loop_
_rlnMicrographName
tilt1_are_Imod/tilt1.mrc
tilt2_are_Imod/tilt2.mrc
tilt3_are_Imod/tilt3.mrc
tilt4_are_Imod/tilt4.mrc
tilt5_are_Imod/tilt5.mrc
tilt6_are_Imod/tilt6.mrc
tilt7_are_Imod/tilt7.mrc
tilt8_are_Imod/tilt8.mrc
tilt9_are_Imod/tilt9.mrc
tilt10_are_Imod/tilt10.mrc
tilt11_are_Imod/tilt11.mrc
tilt12_are_Imod/tilt12.mrc
```

In the ctf reconstruction phase, you should have the picked coordinates, the tilt angle for each tilt in the micrograph with suffix '.tlt', the exposure for each tilt image with suffix '.order', and the defocus for each tilt image estiamted by ctfplotter with name 'ctfplotter.defocus'. The ctfplotter.defocus file is in the micrograph folder. You can then look into the ```relion_ctf_prepare.py``` and set the parameters according to your experimental settings. When all these inputs are ready and the parameters are properly set in ```relion_ctf_prepare.py``` , you can generate the per-particle corrected CTF using 
```
python2 relion_ctf_prepare.py
```
This script will output the starfiles for the CTFs of subtomograms, and also the starfile with paths of all subtomograms, which is specified by ```subtomostarname``` in the script. 

To perform subtomogram averaging using RELION 3.0.8, you should also reconstruct the CTF volume. The python script relion_ctf_prepare.py will output a script name ```do_all_reconstruct_ctfs.sh```, you can reconstruct ctfs using

```
sh do_all_reconstruct_ctfs.sh 128
```
where 128 represents the size of subtomogram. To speed up calculation, you can split the script into multiple files, and execute them independently.

Finally, you can perform subtomogram averaging using RELION and the following command:

```
mpirun -n 7  --oversubscribe  --bind-to none --mca btl '^openib' relion_refine_mpi --o Refine3D/jobdeep/run --auto_refine --split_random_halves --i all_subtomo.star --ref Refine3D/job002/run_class001.mrc --ini_high 40 --dont_combine_weights_via_disc --pool 3 --pad 2  --ctf --particle_diameter 337 --flatten_solvent --zero_mask --oversampling 1 --healpix_order 2 --auto_local_healpix_order 4 --offset_range 6 --offset_step 2 --sym C1 --low_resol_join_halves 40 --norm --scale  --j 4 --free_gpu_memory 256 --gpu 0,1,2,3
```
Now, you should have all necessary files for structural heterogeneity analysis in OPUS-TOMO! Congratulations!

Besides, you may also try prepare all necessary files using the Warp from http://www.warpem.com/warp/! (Sadly, I don't have a windows machine to test this :-()

**Usage Example:**

In overall, the commands for training in OPUS-TOMO can be invoked by calling
```
dsd commandx ...
```
while the commands for result analysis can be accessed by calling
```
dsdsh commandx ...
```

More information about each argument of the command can be displayed using

```
dsd commandx -h 
```
or
```
dsdsh commandx -h
```

**Data Preparation for OPUS-TOMO Using ```dsdsh prepare```:**

There is a command ```dsdsh prepare``` for data preparation. Under the hood, ```dsdsh prepare``` points to the prepare.sh inside analysis_scripts. Suppose **the version of star file is 3.1**, the above process can be simplified as,
```
dsdsh prepare /work/consensus_data.star 236 2.1 --relion31
                $1                      $2    $3    $4
```
 - $1 specifies the path of the starfile,
 - $2 specifies the dimension of subtomogram
 - $3 specifies the angstrom per pixel of subtomogram
 - $4 indicates the version of starfile, only include --relion31 if the file version is higher than 3.0

**The pose pkl can be found in the same directory of the starfile, in this case, the pose pkl is /work/consensus_data_pose_euler.pkl.**

Finally, you should **create a mask using the consensus model and RELION** through ```postprocess```. The detailed procedure for mask creation can be found in https://relion.readthedocs.io/en/release-3.1/SPA_tutorial/Mask.html. 

**Data preparation under the hood**

The pose parameters for subtomograms are stored as the python pickle files, aka pkl. Suppose the refinement result is stored as `consensus_data.star` and **the format of the Relion STAR file is below version 3.0**,
and the consensus_data.star is located at ```/work/``` directory, you can convert STAR to the pose pkl file by executing the command below:

```
dsd parse_pose_star /work/consensus_data.star -D 236 --Apix 2.1 -o ribo-pose-euler.pkl
```
 where

| argument | explanation|
| --- | --- |
| -D | the dimension of the particle subtomogram in your dataset|
| --Apix | is the angstrom per pixel of you dataset|
| -o | followed by the filename of pose parameter used by our program|
| --relion31 | include this argument if you are using star file from relion with version higher than 3.0|


# training <a name="training"></a>

## train_tomo for OPUS-TOMO <div id="train_tomo">

When the inputs are available, you can train the vae for structural disentanglement proposed in OPUS-TOMO's paper using

```
dsd train_tomo /work/ribo.star --poses ./ribo_pose_euler.pkl -n 40 -b 8 --zdim 12 --lr 5.e-5 --num-gpus 4 --multigpu --beta-control 0.8 --beta cos -o /work/ribo -r ./mask.mrc --downfrac 0.65 --valfrac 0.1 --lamb 0.8 --split ribo-split.pkl --bfactor 4. --templateres 160 --angpix 2.1 --estpose --tmp-prefix ref --datadir /work/
```

The argument following train_tomo specifies the starfile for subtomograms. In contrast to OPUS-DSD, we no longer need to specify ctf since they are read from the subtomogram starfile.
Moreover, OPUS-TOMO needs to specify the angpix of the subtomogram by --angpix, and also the prefix directory before the filename for subtomogram in starfile by --datadir /work/ .

The functionality of each argument is explained in the table:
| argument |  explanation |
| --- | --- |
| --poses | pose parameters of the subtomograms in starfile |
| -n     | the number of training epoches, each training epoch loops through all subtomograms in the training set |
| -b     | the number of subtomograms for each batch on each gpu, depends on the size of available gpu memory|
| --zdim  | the dimension of latent encodings, increase the zdim will improve the fitting capacity of neural network, but may risk of overfitting |
| --lr    | the initial learning rate for adam optimizer, 5.e-5 should work fine. |
| --num-gpus | the number of gpus used for training, note that the total number of subtomograms in the total batch will be n*num-gpus |
| --multigpu |toggle on the data parallel version |
| --beta-control |the restraint strength of the beta-vae prior, the larger the argument, the stronger the restraint. The scale of beta-control should be propotional to the SNR of dataset. Suitable beta-control might help disentanglement by increasing the magnitude of latent encodings and the sparsity of latent encodings, for more details, check out [beta vae paper](https://openreview.net/forum?id=Sy2fzU9gl). In our implementation, we adjust the scale of beta-control automatically based on SNR estimation, possible ranges of this argument are [0.5-4.]. You can use larger beta-control for dataset with higher SNR|
| --beta |the schedule for restraint stengths, ```cos``` implements the [cyclic annealing schedule](https://www.microsoft.com/en-us/research/blog/less-pain-more-gain-a-simple-method-for-vae-training-with-less-of-that-kl-vanishing-agony/) and is the default option|
| -o | the directory name for storing results, such as model weights, latent encodings |
| -r | ***the solvent mask created from consensus model***, our program will focus on fitting the contents inside the mask. Since the majority part of subtomogram doesn't contain electron density, using the original subtomogram size is wasteful, by specifying a mask, our program will automatically determine a suitable crop rate to keep only the region with densities. |
| --downfrac | the downsampling fraction of input subtomogram, the input to network will be downsampled to size of D\*downfrac, where D is the original size of subtomogram. You can set it according to resolution of consensus model and the ***templateres*** you set. |
| --lamb | the restraint strength of structural disentanglement prior proposed in OPUS-DSD, set it according to the SNR of your dataset, for dataset with high SNR such as ribosome, splicesome, you can set it to 1. or higher, for dataset with lower SNR, consider lowering it. Possible ranges are [0.1, 3.]. If you find **the UMAP of embeedings is exaggeratedly stretched into a ribbon**, then the lamb you used during training is too high! |
| --split | the filename for storing the train-validation split of subtomograms |
| --valfrac | the fraction of subtomograms in the validation set, default is 0.1 |
| --bfactor | will apply exp(-bfactor/4 * s^2 * 4*pi^2) decaying to the FT of reconstruction, s is the magnitude of frequency, increase it leads to sharper reconstruction, but takes longer to reveal the part of model with weak density since it actually dampens learning rate, possible ranges are [3, 5]. Consider using higher values for more dynamic structures. We will decay the bfactor slightly in every epoch. This is equivalent to learning rate warming up. |
| --templateres | the size of output volume of our convolutional network, it will be further resampled by spatial transformer before projecting to subtomograms. The default value is 192. You may keep it around ```D*downfrac/0.75```, which is larger than the input size. This corresponds to downsampling from the output volume of our network. You can tweak it to other resolutions, larger resolutions can generate sharper density maps, ***choices are Nx16, where N is integer between 8 and 16*** |
| --plot | you can also specify this argument if you want to monitor how the reconstruction progress, our program will display the Z-projection of subtomograms and experimental subtomograms after 8 times logging intervals. Namely, you switch to interative mode by including this. The interative mode should be run using command ```python -m cryodrgn.commands.train_tomo```|
| --tmp-prefix | the prefix of intermediate reconstructions, default value is ```tmp```. OPUS-TOMO will output temporary reconstructions to the root directory of this program when training, whose names are ```$tmp-prefix.mrc``` |
| --angpix | the angstrom per pixel for the input subtomogram |
| --datadir | the root directory before the filename of subtomogram in input starfile |
| --estpose | estimate a pose correction for each subtomogram during training |

The plot mode will ouput the following images in the directory where you issued the training command:

![ref](https://github.com/alncat/opusTomo/assets/3967300/9bd5ce75-c57f-4685-9c34-da55e38fedfe)



Each row shows a selected subtomogram and its reconstruction from a batch.
In the first row, the first image is the projection of experimental subtomogram supplemented to encoder, the second image is a 2D projection from subtomogram reconstruction blurred by the corresponding CTF, the third image is the projection of correpsonding experimental subtomogram after 3D masking.


You can use ```nohup``` to let the above command execute in background and use redirections like ```1>log 2>err``` to redirect ouput and error messages to the corresponding files.
Happy Training! **Open an issue when running into any troubles.**

To restart execution from a checkpoint, you can use

```
dsd train_tomo /work/ribo.star --poses ./ribo_pose_euler.pkl --lazy-single -n 20 --pe-type vanilla --encode-mode grad --template-type conv -b 12 --zdim 12 --lr 1.e-4  --num-gpus 4 --multigpu --beta-control 2. --beta cos -o /work/ribo -r ./mask.mrc --downfrac 0.75 --lamb 1. --valfrac 0.25 --load /work/ribo/weights.0.pkl --latents /work/sp/z.0.pkl --split ribo-split.pkl --bfactor 4. --templateres 160
```
| argument |  explanation |
| --- | --- |
| --load | the weight checkpoint from the restarting epoch |
| --latents | the latent encodings from the restarting epoch |

both are in the output directory

During training, opus-TOMO will output temporary volumes called ```tmp*.mrc``` (or the prefix you specified), you can check the intermediate results by viewing them in Chimera. By default, opus-TOMO reads subotomograms from disk as needed during training.

# analyze result <a name="analysis"></a>
You can use the analysis scripts in ```dsdsh``` to visualize the learned latent space! The analysis procedure is detailed as following and the same as OPUS-DSD!

The analysis scripts can be invoked by calling command like
```
dsdsh commandx ...
```

To access detailed usage information for each command, execute the following:
```
dsdsh commandx -h
```
## sample latent spaces <div id="sample">
The first step is to sample the latent space using kmeans and PCA algorithms. Suppose the training results are in ```/work/sp```, 
```
dsdsh analyze /work/sp 16 4 16
                $1    $2 $3 $4
```

- $1 is the output directory used in training, which stores ```weights.*.pkl, z.*.pkl, config.pkl```
- $2 is the epoch number you would like to analyze,
- $3 is the number of PCs you would like to sample for traversal
- $4 is the number of clusters for kmeans clustering.

The analysis result will be stored in /work/ribo/analyze.16, i.e., the output directory plus the epoch number you analyzed, using the above command. You can find the UMAP with the labeled kmeans centers in /work/ribo/analyze.16/kmeans16/umap.png and the umap with particles colored by their projection parameter in /work/ribo/analyze.16/umap.png .

## reconstruct volumes <div id="reconstruct">
After executing the above command once, you may skip the lengthy umap embedding laterly by appending ```--skip-umap``` to the command in analyze.sh. Our analysis script will read the pickled umap embeddings directly.
The eval_vol command has following options,

You can either generate the volume which corresponds to KMeans cluster centroid or traverses the principal component using,
(you can check the content of script first, there are two commands, one is used to evaluate volume at kmeans center, another one is for PC traversal, just choose one according to your use case)

```
dsdsh eval_vol /work/sp 16 kmeans 16 2.2
                 $1     $2   $3   $4  $5
```

- $1 is the output directory used in training, which stores ```weights.*.pkl, z.*.pkl, config.pkl``` and the clustering result
- $2 is the epoch number you just analyzed
- $3 specifies the kind of analysis result where volumes are generated, i.e., kmeans clusters or principal components, use ```kmeans``` for kmeans clustering, or ```pc``` for principal components
- $4 is the number of kmeans clusters (or principal component) you used in analysis
- $5 is the apix of the generated volumes, you can specify a target value

change to directory ```/work/sp/analyze.16/kmeans16``` to checkout the reference*.mrc, which are the reconstructions
correspond to the cluster centroids.

You can use

```
dsdsh eval_vol /work/sp 16 pc 1 2.2
                $1      $2 $3 $4 $5
```


to generate volumes along pc1. You can check volumes in ```/work/sp/analyze.16/pc1```. You can make a movie using chimerax's ```vseries``` feature. An example script for visualizing movie is in ```analysis_scripts/movie.py```. You can show the movie of the volumes by ```ChimeraX --script "./analysis_scripts/movie.py reference 0.985```.
**PCs are great for visualizing the main motions and compositional changes of marcomolecules, while KMeans reveals representative conformations in higher qualities.**
To invert the handness of the reconstrution, you can include ```--flip``` to the above commands.

## select particles <div id="select">
Finally, you can also retrieve the star files for subtomograms in each kmeans cluster using

```
dsdsh parse_pose /work/consensus_data.star 320 1.699 /work/sp 16 16 --relion31
                                $1           $2  $3    $4     $5 $6  $7
```

- $1 is the star file of all subtomograms
- $2 is the dimension of subtomogram
- $3 is apix value of subtomogram
- $4 is the output directory used in training
- $5 is the epoch number you just analyzed
- $6 is the number of kmeans clusters you used in analysis
- $7 indicates the version of starfile, only include this when the version of starfile is higher than 3.0

change to directory ```/work/sp/analyze.16/kmeans16``` to checkout the starfile for subtomograms in each cluster.

This program is built upon a set of great works:
- [opusDSD](https://github.com/alncat/opusDSD)
- [cryoDRGN](https://github.com/zhonge/cryodrgn)
- [Neural Volumes](https://stephenlombardi.github.io/projects/neuralvolumes/)
- [UMAP](https://umap-learn.readthedocs.io/en/latest/)
- [Healpy](https://healpy.readthedocs.io/en/latest/)
