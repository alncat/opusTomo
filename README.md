# Table of contents
1. [Opus-TOMO](#opustomo)
2. [setup environment](#setup)
3. [training](#training)
   1. [train_tomo](#train_tomo)
4. [analyze result](#analysis)
   1. [sample latent spaces](#sample)
   2. [reconstruct volumes](#reconstruct)
   3. [select particles](#select)

# Opus-TOMO <div id="opustomo">
This repository contains the implementation of opus-tomography (OPUS-TOMO), which is developed by Zhenwei (Benedict，本笃) Luo at the group of
Prof. Jianpeng Ma at Fudan University. OPUS-TOMO works with **WARP/M** pipeline in a seamless way to facilitate high-resolution cryo-ET structure determination and reveal dynamic process for macromolecule in situ! Specifically, it can **filter template matching result to obtain highly homogeneous subtomogram set that achieves sub-nanometer resolution**. Secondly, it can **reconstruct dynamics in various scales from subtomogram averaging (STA) results** from **M**! OPUS-TOMO can even work with tomogram reconstructed by **AreTOMO**. The tutorials are in https://github.com/alncat/opusTomo/wiki!
The preprint of OPUS-TOMO is available at https://www.biorxiv.org/content/10.1101/2024.06.30.601442 or https://drive.google.com/drive/folders/1FcF1PC-0lY2C6DP0zP7K7qhcz_ltn5t-?usp=sharing.

Structural heterogeneity is a central problem for cryo-ET which occurs at many stage of tomography processing. Specifically, at the very first stage, after reconstructing a tomogram, you then encounter the problem to pick subtomograms corresponding to macromolecule of interest from the tomogram. Given the abundance of different molecules inside the
cell sample, subtomogram picking faces large challenges from structural heterogeneity. At a later stage, when you have obtained a purer set of subtomograms for the molecule of interest, you may still encounter the structural heterogeneity problem as the molecule is in constant dynamics in cell environment. 
The vitrified samples then preserves lots of different conformations and compositions of the molecule of interest.

The main functionality of OPUS-TOMO is reconstructing conformational and compositonal changes at all kinds of stages from cryo-ET data end-to-end! OPUS-TOMO can be applied to data analysis at any stage, even when you are just picking particles. Using the template matching result with determined pose from **pyTOM**, 
OPUS-TOMO can be used to filter picked particles! At later stage, OPUS-TOMO can not only disentangle 3D structural information by reconstructing different compositions, but also reconstruct continous conformational dynamics for the macromolecules in cellular environment. 

Exemplar dynamics resolved by OPUS-TOMO is shown below:
The elementary step of ATP synthesis by 120 degrees rotation of the central stalk and the 30 degrees rotation of the F_1 Head,

https://github.com/user-attachments/assets/4cbb29f6-f1fa-48bb-a555-50845fe1b50e

For S. pombe 80S ribosome, a part of translocation dynamics resolved by traversing PC3, which shows the translocation of A/T- and P- site tRNAs to A/P- and P/E- site tRNAs.
A superb reference for the translation elongation cycle can be found in Ranjan's work, https://www.embopress.org/doi/full/10.15252/embj.2020106449 .

https://github.com/user-attachments/assets/231ac7b9-0f43-4234-8617-49c3f3e30aee


A part of translation elongation cycle resolved by traversing PC9, which shows the translocation of A/T- tRNAs to A-site and the exit of E-site tRNA for Cm-treated M. pneumoniae 70S ribosome.
The model is trained using M refined subtomograms.

https://github.com/user-attachments/assets/ea9d12d1-29f0-4572-b1d4-3c22d1c77d89


Another important function of OPUS-TOMO is clustering the template matching result and achiving high-resolution reconstruction! You can check the wiki pages for a tutorial (https://github.com/alncat/opusTomo/wiki)
<img width="1045" height="612" alt="image" src="https://github.com/user-attachments/assets/b09ccb2d-8931-4119-b185-a571b06c0571" />

<img width="778" height="573" alt="image" src="https://github.com/user-attachments/assets/4ef8ed14-26ea-435d-86a5-85ef0e860dbf" />


The histograms shows that OPUS-TOMO can retrieve subtomograms from Template Matching result that overlap with the ground-truth results from DeePiCt.
TM Overlapping With DeePiCt refers to the number of subtomograms from Template Mathching that overlap with DeePiCt's result, and OPUS-TOMO
refers to the number of subtomograms in classes 12-17 that overlaps with DeePiCt's result. TM Overlapping Ratio refers to the ratio of subtomograms that overlaps with 
DeePiCt's result in each tilt series, and OPUS-TOMO refers to the ratio of subtomograms in classes 12-17 that overlaps with DeePiCt's result in each tilt series. It is worth noting that OPUS-TOMO is trained by using the subtomograms with their 
orientation determined by template matching in pyTOM directly without any other processing. The template matching results for this case are in the folder https://drive.google.com/drive/folders/1xR_zD_nF9Hvw9S3nsjxxPR2DQmod3Fmu?usp=drive_link .

The workflow of OPUS-TOMO is demonstrated as follows:
<img width="653" height="301" alt="image" src="https://github.com/user-attachments/assets/9a20e483-6f9e-48b8-9ac9-b18c8ae1d289" />

Note that all input and output of this method are in real space! 
The architecture of encoder is (Encoder class in cryodrgn/models.py):

<img width="2056" height="314" alt="image" src="https://github.com/user-attachments/assets/641c9cb1-4c8e-4eb2-a2e2-a2b9cb3e4875" />

The architecture of decoder is (ConvTemplate class in cryodrgn/models.py. In this version, the default size of output volume is set to 192^3, I downsampled the intermediate activations to save some gpu memories. You can tune it as you wish, happy training!):

<img width="964" alt="image" src="https://github.com/user-attachments/assets/ed448e4a-3097-473c-8d2a-d50725e1c735">

The architecture of dynamics decoder is:

<img width="354" height="106" alt="image" src="https://github.com/user-attachments/assets/dbbe08df-210f-465d-a3e6-94dac417f815" />

OPUS-TOMO takes the **3D subtomograms** as input directly. The capacity of OPUS-TOMO is **robust against subtomograms from all kinds of particle localization methods**, such as neural-network based DeePiCt (https://github.com/ZauggGroup/DeePiCt), and the most crude templated matching in PyTom (https://github.com/SBC-Utrecht/PyTom/, this implementation is GPU-accelerated with a user-friendly GUI!). OPUS-TOMO also enables the structural heterogeneity anlaysis using **the simplest statistic methods, PCA and KMeans clustering**. These two approaches can lead to sufficiently rich discovery about the conformational and compositional changes of macromolecules. Specifically, **PCA** in learned latent space allows decomposing structural variations of macromolecules in cryo-ET dataset into dinstinct modes that greatly facilitate reserchers' understandings. This is in a similar spirit to the Normal Mode Analysis (NMA) for macromolecules which investigates their movement modes.


## S.pombe 80S Ribosome <a name="80s"></a>
The S.pombe dataset is publicly available at EMPIAR-10988 (https://www.ebi.ac.uk/empiar/EMPIAR-10988/).
In this dataset, OPUS-TOMO has shown superior structural disentanglement ability to capture continous structral changes into PCs of the composition latent space, and characterizes 
functionally important subpopulations. The results are deposited in https://zenodo.org/records/12631920.

<img width="1079" height="617" alt="image" src="https://github.com/user-attachments/assets/f329eefb-fe80-40a6-a98c-8b6f797eeb9b" />



## FAS <a name="fas"></a>

It can even reconstruct higher resolution structure for FAS in EMPIAR-10988 by clustering 221 particles from 4800 noisy subtomograms picked by template matching!
The template matching and subtomogram averaging results are in the folder https://drive.google.com/drive/folders/1OijHVrCu3M-OgqvNu_YZ4jW8OWn6OwaV?usp=drive_link, fasp_expanded.star stores the subtomogram averaging results after D3 symmetry expansion.

We can also reconstruct the dynamics for FAS, using only 221 particles!

https://github.com/user-attachments/assets/806c518c-427d-41c9-905d-17b18fba8922

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
Moreover, OPUS-TOMO needs to specify the angpix of the subtomogram by ```--angpix```, and also the prefix directory before the filename for subtomogram in starfile by ```--datadir```.

The functionality of each argument is explained in the table:
| argument |  explanation |
| --- | --- |
| --poses | pose parameters of the subtomograms in starfile |
| -n     | the number of training epoches, each training epoch loops through all subtomograms in the training set |
| -b     | the number of subtomograms for each batch on each gpu, depends on the size of available gpu memory|
| --zdim  | the dimension of latent encodings, increase the zdim will improve the fitting capacity of neural network, but may risk of overfitting |
| --zaffinedim | the dimension of dynamics latent encodings, default is 4 |
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
| --bfactor | will apply exp(-bfactor/4 * s^2 * 4*pi^2/ angpix**2 ) decaying to the FT of reconstruction, s is the magnitude of frequency, increase it leads to sharper reconstruction, but takes longer to reveal the part of model with weak density since it actually dampens learning rate, possible ranges are [3, 5]. Consider using higher values for more dynamic structures.  |
| --templateres | the size of output volume of our convolutional network, it will be further resampled by spatial transformer before projecting to subtomograms. The default value is 192. You may keep it around ```D*downfrac/0.75```, which is larger than the input size. This corresponds to downsampling from the output volume of our network. You can tweak it to other resolutions, larger resolutions can generate sharper density maps, ***choices are Nx16, where N is integer between 8 and 16*** |
| --plot | you can also specify this argument if you want to monitor how the reconstruction progress, our program will display the Z-projection of subtomograms and experimental subtomograms after 8 times logging intervals. Namely, you switch to interative mode by including this. The interative mode should be run using command ```python -m cryodrgn.commands.train_tomo```|
| --tmp-prefix | the prefix of intermediate reconstructions, default value is ```tmp```. OPUS-TOMO will output temporary reconstructions to the root directory of this program when training, whose names are ```$tmp-prefix.mrc``` |
| --angpix | the angstrom per pixel for the input subtomogram |
| --datadir | the root directory before the filename of subtomogram in input starfile |
| --estpose | estimate a pose correction for each subtomogram during training |
| --masks | the mask parameters for defining the rigid-body dynamics |
| --ctfalpha | the degree of ctf correction to experimental subtomogram, the default value is 0, which is equivalent to phase flipping. You can also try value like 0.5, which will further correct the amplitude of FT of subtomogram by the square root of the ctf function|
| --ctfbeta | the degree of ctf correction to the reconstruction output by decoder, the default value is 1. You can also try value like 0.5, which will make the voxel values output by decoder smaller since the ctf correction is weaker.|
| --tilt-step | the interval between successive tilts, default is 2, you need to change it according to your experimental settings when training subtomograms from WARP|
| --tilt-range | the range of tilt angles, default is 50, change it to your range when training subtomograms from WARP|

The plot mode will ouput the following images in the directory where you issued the training command, the fllowing images are for ATP synthase dimer:

<img width="441" height="331" alt="image" src="https://github.com/user-attachments/assets/e19105e7-34fa-409b-8cbc-2abf2e870e87" />

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

change to directory ```/work/sp/analyze.16/kmeans16``` to checkout the starfile for subtomograms in each cluster. With those starfiles, you can easily obtain
the unfiltered reconstruction using relion via the command
```
relion_reconstruct --i pre1.star --3d_rot
```

This program is built upon a set of great works:
- [opusDSD](https://github.com/alncat/opusDSD)
- [cryoDRGN](https://github.com/zhonge/cryodrgn)
- [Neural Volumes](https://stephenlombardi.github.io/projects/neuralvolumes/)
- [UMAP](https://umap-learn.readthedocs.io/en/latest/)
- [Healpy](https://healpy.readthedocs.io/en/latest/)

The content below is just for backup purpose, which may not be compitable with the latest version of OPUS-TOMO.
# prepare data <a name="preparation"></a>

**Data Preparation Guidelines:**
1. **Cryo-ET Dataset:** The form of inputs of OPUS-TOMO is similar to the inputs required for subtomogram averaging in Relion 3.0.8, which consists of subtomograms and the CTF parameters of tilts for each subtomogram in a starfile. Ensure that the cryo-ET dataset is stored as separate subtomograms in directory. A good dataset for tutorial is the S.pombe which is available at https://empiar.pdbj.org/entry/10180/ (It contains the coordinates for subtomograms and tilt alignment parameters for reconstructing tomograms.) We also have a script adapted from Relion for preparing the subtomograms and ctf starfiles. It is named as *relion_ctf_prepare.py* in the root folder of this repository or in the google drive https://drive.google.com/drive/folders/1FcF1PC-0lY2C6DP0zP7K7qhcz_ltn5t-?usp=sharing. You can follow the detailed data preparation process in this paper: https://www.nature.com/articles/nprot.2016.124 or the instruction below.

2. **Subtomogram averaging/Template matching Result:** The program requires a subtomogram averaging or template matching results, where the pose of subtomogram is determined, which should not apply any symmetry and must be stored as a Relion STAR file.

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
TS_026_st.mrc is the tomogram, which can be reconstrcuted using AreTOMO with the script ```recon.sh``` which was shared in https://drive.google.com/drive/folders/1FcF1PC-0lY2C6DP0zP7K7qhcz_ltn5t-?usp=sharing, (Link the tomogram to this directory and this name! It's recommended to use AreTOMO with commit hash f7d1d44c6cb68756be4776b4c3a79a02f491e278, you can checkout this commit using ```git checkout f7d1d44c6cb68756be4776b4c3a79a02f491e278``` This version of AreTOMO can produce all required files for working with IMOD). TS_026_st.mrcs is the tilt series stack, TS_026_st.ali is the aligned tilt series stack (mainly used for CTF estimation),
TS_026_st.xf is the alignment parameter for IMOD, TS_026_st.tlt records the tilt angles for each tilt in tilt series stack, 
TS_026_st.order records the tilt angle and its corresponding exposure dose (You can create a fake one with exposure dose being the index of tilt frame). For example,
```
-30 15.00000000000000000000
-28 14.00000000000000000000
-26 13.00000000000000000000
-24 12.00000000000000000000
-22 11.00000000000000000000
-20 10.00000000000000000000
-18 9.00000000000000000000
-16 8.00000000000000000000
-14 7.00000000000000000000
-12 6.00000000000000000000
-10 5.00000000000000000000
-8 4.00000000000000000000
-6 3.00000000000000000000
-4 2.00000000000000000000
-2 1.00000000000000000000
0 0
2 1.00000000000000000000
4 2.00000000000000000000
6 3.00000000000000000000
8 4.00000000000000000000
10 5.00000000000000000000
12 6.00000000000000000000
14 7.00000000000000000000
16 8.00000000000000000000
18 9.00000000000000000000
20 10.00000000000000000000
22 11.00000000000000000000
24 12.00000000000000000000
26 13.00000000000000000000
28 14.00000000000000000000
30 15.00000000000000000000
```
TS_026_st.coords records the coordinate of subtomograms (which is picked by DeePiCt or PyTOM). PyTOM features a user-friendly GUI! 
For PyTOM template matching result, you need to first convert the particle picking result in xml file to star file using ```conv.sh``` in the shared google drive folder.
The starfile then contains the orientation of each subtomogram, which allows further filtering using OPUS-TOMO. An example starfile is of the form,
```
data_

loop_
_rlnCoordinateX #1
_rlnCoordinateY #2
_rlnCoordinateZ #3
_rlnMicrographName #4
_rlnMagnification #5
_rlnDetectorPixelSize #6
_rlnGroupNumber #7
_rlnAngleRot #8
_rlnAngleTilt #9
_rlnAnglePsi #10
```

You also need to convert the starfile to **coords** file using the scrip "star2csv.py" in the shared google drive folder (Pay attention to the bin size of tomogram where subtomograms are extracted, make sure the
 the scale of final coordinate match with the size of tomogram using ```factor``` variable in the script). 
You can then move the **coords** files to the respective tilt series micrograph folders.
Finally, you need to combine the starfile for each tilt series into a single starfile.

In the subtomogram extraction phase, you should have the picked coordinates (from PyTOM or DeePiCt), and tomograms. Put the coordinate files into the same directory as the tomogram. The subtomograms can be extracted by RELION 3.0.8 using the command
```
relion_preprocess --coord_suffix .coords --coord_dir ./ --part_dir Extract/extract_tomo/ --extract --bg_radius 44 --norm --extract_size 128 --i all_tomograms.star
```
where --coord_suffix specifies the suffix of coordinate files, --coord_dir ./ tells relion_preprocess that the coords files are in the same folder as the tomogram, --part_dir specifies the output directory for subtomograms, --bg_radius specifies the radius for circle mask, --extract_size specifies the size of subtomogram, and --i specifies the starfile with the path of tomograms, which has the following contents:
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

In the ctf reconstruction phase, you should have the picked coordinates, the tilt angle for each tilt in the tilt series with suffix '.tlt' (This will be generated by AreTOMO automatically), the exposure for each tilt image with suffix '.order', and the defocus for each tilt image estiamted by ctfplotter with name 'ctfplotter.defocus'. The ctfplotter.defocus file is in the micrograph folder. The ctf estimation is done by **ctfplotter**. We provided an example configuration file for **ctfplotter** which is named as ctfplotter.com in https://drive.google.com/drive/folders/1FcF1PC-0lY2C6DP0zP7K7qhcz_ltn5t-?usp=sharing. 
Firstly, we need to create an aligned stack in the micrograph directory using **newstack** from IMOD, ```newstack -pa newst.com```, which will generate a file with suffix **.ali**, e.g., TS_026_st.ali.
The ctf estimation can then be performed by ```ctfplotter -pa ctfplotter.com```. An example defocus file is of the form,
```
1       0       0.0     0.0     0.0     3
40      40      -60.00  -60.00   3416.3  3248.6   88.22
39      39      -57.00  -57.00   3632.9  3484.6   88.70
36      36      -54.00  -54.00   3931.3  3782.7   87.20
35      35      -51.00  -51.00   3823.0  3746.1  -65.90
32      32      -48.00  -48.00   3799.6  3776.4  -39.16
31      31      -45.00  -45.00   3783.6  3750.9  -38.99
28      28      -42.00  -42.00   3805.5  3776.0  -24.21
27      27      -39.00  -39.00   3821.9  3797.4  -17.51
24      24      -36.00  -36.00   3815.9  3808.4   -7.63
23      23      -33.00  -33.00   3840.8  3833.9  -15.56
20      20      -30.00  -30.00   3817.2  3805.1  -60.19
19      19      -27.00  -27.00   3828.3  3814.3  -74.87
16      16      -24.00  -24.00   3818.8  3808.5  -82.24
15      15      -21.00  -21.00   3827.9  3811.8  -84.89
12      12      -18.00  -18.00   3846.8  3797.1  -87.33
11      11      -15.00  -15.00   3844.1  3784.1   86.43
```
where, the last four columns are tilt angle, defocus u, defocus v, astigmatism, respectively.

When defoci of all tilt series are estimated. You can then have a look at the ```relion_ctf_prepare.py``` and set the parameters according to your experimental settings (**There is a version compatible with python3 in the google drive**). When all these inputs are ready and the parameters are properly set in ```relion_ctf_prepare.py``` , you can generate the per-particle corrected CTF using 
```
python relion_ctf_prepare.py
```
This script will output the starfiles for the CTFs of subtomograms, and the starfile with all subtomograms' locations. 

**For template matching result from PyTOM, you need to combine the starfile generated by ```relion_ctf_prepare.py``` and the starfile converted by pyTOM which 
contains subtomogram ortientations.** You can do this by assigning the ```rlnAngleRot, rlnAngleTilt, rlnAnglePsi``` columns in the starfile for template matching result
to the starfile generated by ```relion_ctf_prepare.py```. The operations on starfile can be conveniently handled by starpy at https://github.com/fuzikt/starpy. This starfile with 
all orientation information can be used to train OPUS-TOMO! The block below shows the necessary columns in the combined starfile.
```
data_

loop_
_rlnMicrographName #1
_rlnCoordinateX #2
_rlnCoordinateY #3
_rlnCoordinateZ #4
_rlnImageName #5
_rlnCtfImage #6
_rlnMagnification #7
_rlnDetectorPixelSize #8
_rlnAngleRot #9
_rlnAngleTilt #10
_rlnAnglePsi #11
```

**Correctness check**:

Opus-TOMO mainly assumes the tomogram is reconstructed by the weighted backprojection (WBP) algorithm in Aretomo. The underlying principle for WBP can be referred to the note (https://static1.squarespace.com/static/56b6357e01dbaea0266fe701/t/56edae5237013bc012524a8d/1458417249296/Tomography+Interpreted+as+a+Filtered+Back+Projection+-+Martin+Nilsson.pdf#page=17.19), which gives the detailed mathamtical derivation for inverse Radon transform. 
The tilt image will be first multiplied with $`cos(\theta)`$, where $`\theta`$ is the tilt angle, to account the tilt-dependent effective sample thickness.
The fourier transform of tilt image will be multiplied with R-weight (r or $`r*(0.55 + 0.45 cos(2\pi r)`$) in Aretomo, where r is the modulus of frequency vector), then performed an inverse fourier transform and backproject according to its tilt angle. Aretomo's output has the same convention as IMOD's though Aretomo goes through a series of transformations.
It is worthing noting that the tilt angle for CTF is inverted in ```relion_ctf_prepare.ctf``` to accommodate Relion's backprojection convention. Aretomo's backprojection convention can be checked in their source code here: https://github.com/czimaginginstitute/AreTomo2/blob/e9f89413352511f6cac0974a93fd6ff21b9c4129/Recon/GBackProj.cu#L31). 
This convention will make negative tilt angle locate at positive z axis.
Aretomo reconstruct tomogram in real space. Moreover, it flipped z axis after backprojection to make the tomogram match with IMOD's convention. Check
https://github.com/czimaginginstitute/AreTomo2/blob/e9f89413352511f6cac0974a93fd6ff21b9c4129/Recon/CDoWbpRecon.cpp#L130

The 3DCTF reconstruction is performed by compute_3dctf in ```ctf.py``` in opus-TOMO. You can check the rot_2d in ```lie_tools.py``` to learn the notation of rotation in opus-TOMO. 
OPUS-TOMO make positive x axis rotate to negative z axis under negative tilt angle, which can be expressed as $`x' = R(\theta_{Imod})x`$, where x' is the coordinate after tilting, R is the rotation matrix using normal definition as in ```lie_tools.rot_2d```, and x is the original coordinate in tomogram. (the tilt angle is inverted once again in ```ctf.py```, so it revert to the original tilt angle in IMOD's convention). However, the defocus gradient, i.e., which part of x axis has higher underfocus with negative tilt angle, might be trickier! (You can check whether the negtive x axis or positive x axis swings to higher underfocus under negative tilt angle by estimating the defoci of left or right part. If the negative x axis swings to higher underfocus, then positive z axis has larger underfocus, vice versa. I am preparing a script for checking this!) 
Lastly, the correctness of 3DCTF can be checked by saving the 3DCTF reconstruction, and compare its missing wedge w.r.t the missing wedge of extracted subtomograms. Make sure they look similar! Uncomment these lines https://github.com/alncat/opusTomo/blob/3a6b4efb51d57aa8e8108a729c87f8a2e0555526/cryodrgn/models.py#L707 https://github.com/alncat/opusTomo/blob/77c91475ade5e828b07646ae8fdcdee151572314/cryodrgn/models.py#L1816 to write out fourier transform of subtomogram and CTF reconstructions.

<img width="450" alt="image" src="https://github.com/alncat/opusTomo/assets/3967300/70c7703b-6676-44fc-9b6c-5ccdac7736f5">

This is the fourier transform of a subtomogram and the corresponding 3DCTF for TS_041 in S.pombe dataset. You can see that the missing wedges of these two match. 

The per-particle 3DCTF correction implemented in relion_ctf_prepare.py is very rudimentary. I am preparing a more accurate per-particle 3DCTF correction now.

To perform subtomogram averaging using RELION 3.0.8, you should also reconstruct the CTF volume (though this is not required for training opusTOMO). The python script relion_ctf_prepare.py will output a script name ```do_all_reconstruct_ctfs.sh```, you can reconstruct ctfs using

```
sh do_all_reconstruct_ctfs.sh 128
```
where 128 represents the size of subtomogram. To speed up calculation, you can split the script into multiple files, and execute them independently. 
OpusTOMO reads the starfile for the CTF of subtomogram and reconstruct a 3DCTF ad hoc. Hence, you can also use the per-particle CTF estimated by other programs as long as it is stored as a per-particle STAR file.

Finally, you can perform subtomogram averaging using RELION and the following command:

```
mpirun -n 7  --oversubscribe  --bind-to none --mca btl '^openib' relion_refine_mpi --o Refine3D/jobdeep/run --auto_refine --split_random_halves --i all_subtomo.star --ref Refine3D/job002/run_class001.mrc --ini_high 40 --dont_combine_weights_via_disc --pool 3 --pad 2  --ctf --particle_diameter 337 --flatten_solvent --zero_mask --oversampling 1 --healpix_order 2 --auto_local_healpix_order 4 --offset_range 6 --offset_step 2 --sym C1 --low_resol_join_halves 40 --norm --scale  --j 4 --firstiter_cc --free_gpu_memory 256 --gpu 0,1,2,3
```
Now, you should have all necessary files for structural heterogeneity analysis in OPUS-TOMO! Congratulations!

