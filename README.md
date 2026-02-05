# Table of contents
1. [OPUS-ET](#opustomo)
2. [setup environment](#setup)
3. [training](#training)
   1. [train_tomo](#train_tomo)
4. [analyze result](#analysis)
   1. [sample latent spaces](#sample)
   2. [reconstruct volumes](#reconstruct)
   3. [select particles](#select)
   4. [Interactive filtering with cryoDRGN_filtering_template.ipynb](#filtering)
5. [OPUS-ET Analysis Skill for Kimi Code CLI](#skill)

# OPUS-ET <div id="opustomo">

This repository contains the implementation of OPUS-Electron Tomography (OPUS-ET), developed by Zhenwei (Benedict, 本笃) Luo in the group of Prof. Jianpeng Ma at Fudan University.

OPUS-ET is designed to work seamlessly with the WARP/M pipeline (through a modified WARP which can export subtomogram without CTF correction and CTF parameters as csv, https://github.com/alncat/warp/tree/alncat) to facilitate high-resolution cryo-electron tomography (cryo-ET) structure determination and to reveal in situ macromolecular dynamics. OPUS-ET supports different parallel strategies for efficient training! Note that OPUS-ET is still undergoing code improvements, you can kept up to data by ```git pull```.

▶ Tutorials: see the OPUS-ET wiki:
https://github.com/alncat/opusTomo/wiki

▶ Preprint:
https://www.biorxiv.org/content/10.1101/2025.11.21.688990v1

## Overview

Structural heterogeneity is a central challenge in cryo-ET and arises at multiple stages of the processing pipeline:

1.	Subtomogram picking:
   
After tomogram reconstruction, you need to pick subtomograms corresponding to the macromolecule of interest. Because cells contain a large variety of different molecular species, template matching and subtomogram picking are heavily affected by structural and compositional heterogeneity.

2.	Downstream analysis:
   
Even after obtaining a relatively “pure” set of subtomograms for a target complex, structural heterogeneity persists. Macromolecules in the cellular environment are dynamic and can adopt multiple conformations and compositions. The vitrified samples therefore preserve a rich ensemble of states rather than a single static structure.

OPUS-ET is designed to tackle this structural heterogeneity end-to-end, from the earliest picking stages through to detailed conformational landscape analysis.

## Key Features

1. Filtering template matching results

OPUS-ET can filter template matching outputs (PyTOM) to obtain highly homogeneous subtomogram sets that are suitable for sub-nanometer resolution reconstruction.

2. Multi-scale heterogeneity analysis

OPUS-ET can be applied to STA results (e.g., from M): disentangle compositional heterogeneity, reconstructing different subcomplexes or binding states; reconstruct continuous conformational dynamics, providing a low-dimensional representation of in situ flexibility.

In short, OPUS-ET aims to reconstruct both compositional and conformational changes across the entire cryo-ET processing pipeline, from raw picking to high-resolution structural and dynamical analysis.


Exemplar dynamics resolved by OPUS-ET are shown below:
The counter rotation between F0 and F1 subcomplexes in ATP synthase when switching primary states (3C->1A),

https://github.com/user-attachments/assets/0a743ce3-e09c-4610-a9b4-c355154f856f

For S. pombe 80S ribosome, a part of translocation dynamics resolved by traversing PC3, which shows the translocation of A/T- and P- site tRNAs to A/P- and P/E- site tRNAs.
A superb reference for the translation elongation cycle can be found in Ranjan's work, https://www.embopress.org/doi/full/10.15252/embj.2020106449 .

https://github.com/user-attachments/assets/19db80d0-7de6-4ae6-9549-ead16f61915a

A part of translation elongation cycle resolved by traversing PC9, which shows the translocation of A/T- tRNAs to A-site and the exit of E-site tRNA for Cm-treated M. pneumoniae 70S ribosome.
The model is trained using M refined subtomograms.

https://github.com/user-attachments/assets/9d5a52f7-95dd-4c30-b5bd-938dbfaa5d69

Another important function of OPUS-ET is clustering the template matching result and achiving high-resolution reconstruction, which are detailed in the wiki (https://github.com/alncat/opusTomo/wiki)

<img width="859" height="664" alt="image" src="https://github.com/user-attachments/assets/9f49fb93-e19e-4162-9af4-a198e1e7c9a7" />

The architecture of OPUS-ET is demonstrated as follows:

<img width="838" height="334" alt="image" src="https://github.com/user-attachments/assets/97a52494-18fc-4e12-9314-9fd23a9bd6e8" />

The architecture of encoder is (Encoder class in cryodrgn/models.py):

<img width="2056" height="314" alt="image" src="https://github.com/user-attachments/assets/641c9cb1-4c8e-4eb2-a2e2-a2b9cb3e4875" />

The architecture of composition decoder is (ConvTemplate class in cryodrgn/models.py. In this version, the default size of output volume is set to 160^3, which can be set via ```--templateres```:

<img width="964" alt="image" src="https://github.com/user-attachments/assets/ed448e4a-3097-473c-8d2a-d50725e1c735">

The architecture of conformation decoder is:

<img width="457" height="144" alt="image" src="https://github.com/user-attachments/assets/a34ab59d-fb4e-47c6-9aa7-bd0138661bdb" />

OPUS-ET directly takes 3D subtomograms as input. Its performance is robust across subtomograms obtained from a wide range of particle localization methods, including neural network–based approaches such as DeePiCt (https://github.com/ZauggGroup/DeePiCt￼) and more classical, template-based methods such as PyTom (https://github.com/SBC-Utrecht/PyTom/￼, which provides GPU-accelerated template matching with a user-friendly GUI).

For structural heterogeneity analysis, OPUS-ET incorporates simple yet powerful statistical tools—principally PCA and k-means clustering. In practice, these methods provide rich insights into both conformational and compositional variability within macromolecular assemblies. In particular, applying PCA to the learned latent space enables OPUS-ET to decompose structural variations in cryo-ET datasets into interpretable modes of motion. This greatly facilitates downstream biological interpretation. Conceptually, this latent-space PCA is analogous to normal mode analysis (NMA) in structural biology, which characterizes the intrinsic dynamical modes of macromolecules.

## C. reinhardtii ATP synthase <a name="atp"></a>
The C. reinhardtii dataset is publicly available at EMPIAR-11830. OPUS-ET resolved the rotary substates of ATP synthase in situ.
<img width="792" height="711" alt="image" src="https://github.com/user-attachments/assets/9f5b9e9d-269e-441b-84b9-2c02a76e9e87" />

## S. pombe 80S Ribosome <a name="80s"></a>
The S. pombe dataset is publicly available at EMPIAR-10988 (https://www.ebi.ac.uk/empiar/EMPIAR-10988/).
In this dataset, OPUS-ET has shown superior structural disentanglement ability to capture continous structral changes into PCs of the composition latent space, and characterizes 
functionally important subpopulations. The results are deposited in https://zenodo.org/records/12631920.

<img width="860" height="662" alt="image" src="https://github.com/user-attachments/assets/a13ee6d1-f614-43fb-96b6-6d53f0612e6d" />

## S. pombe FAS <a name="fas"></a>

It can even reconstruct higher resolution structure for FAS in EMPIAR-10988 by clustering 221 particles from 4800 noisy subtomograms picked by template matching!
The template matching and subtomogram averaging results are in the folder https://drive.google.com/drive/folders/1OijHVrCu3M-OgqvNu_YZ4jW8OWn6OwaV?usp=drive_link, fasp_expanded.star stores the subtomogram averaging results after D3 symmetry expansion.

We can also reconstruct the dynamics for FAS, using only 221 particles!

https://github.com/user-attachments/assets/806c518c-427d-41c9-905d-17b18fba8922

# set up environment <a name="setup"></a>

After cloning the repository, to run OPUS-ET, you need to have an environment with pytorch installed and a machine with GPUs. The recommended hardware configuration is a machine with 4 V100 GPUs.
You can create the conda environment for OPUS-ET using the environment file in the source folder by executing

```
conda env create --name opuset -f environment.yml
```

This will create an environment with cuda 11.3 and pytorch 1.11.0. There are also other environment files for choosing. OPUS-ET has several different training scripts implementing different parallelisms. ```dsd train_tomo``` implements a legacy data parallel training. ```cryodrgn.commands.train_tomo_dist``` implements distributed data parallel training using ```torch.distributed```. ```dsd train_tomo_hvd``` implements distributed data parallel training using horovod, which should have horovod installed according to the tutorial https://github.com/alncat/opusTomo/wiki/horovod-installation. After the environment is sucessfully created, you can then activate it and install OPUS-ET within the environment.

```
conda activate opuset
```

You can then install OPUS-ET by changing to the directory with cloned repository, and execute
```
pip install -e .
```

OPUS-ET can be kept up to date by 
```
git pull
```

**Usage Example:**

In overall, the commands for training in OPUS-ET can be invoked by calling
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

**Data Preparation for OPUS-ET Using ```dsdsh prepare```:**

There is a command ```dsdsh prepare``` for data preparation. Under the hood, ```dsdsh prepare``` points to the prepare.sh inside analysis_scripts. Suppose **the version of Relion star file below 3.0**, the data preparation can be done by,
```
dsdsh prepare /work/consensus_data.star 236 2.1
                $1                      $2    $3
```
 - $1 specifies the path of the starfile,
 - $2 specifies the dimension of subtomogram
 - $3 specifies the angstrom per pixel of subtomogram

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

## train_tomo for OPUS-ET <div id="train_tomo">

When the subtomograms and ctfs exported by WARP are available, you can train OPUS-ET using

```
dsd train_tomo /work/ribo.star --poses ./ribo_pose_euler.pkl -n 40 -b 12 --zdim 12 --lr 4.e-5 --num-gpus 4 --multigpu --beta-control 0.5 -o /work/ribo -r ./mask.mrc --downfrac 0.9 --valfrac 0.1 --lamb 0.5 --split ribo-split.pkl --bfactor 3.5 --templateres 128 --angpix 3.37 --estpose --tmp-prefix ref --datadir /work/ --warp --tilt-range 50 --tilt-step 2 --ctfalpha 0. --ctfbeta 1.
```
OPUS-ET provides a ```--float16``` option, which converts input subtomograms to 16-bit floating point (float16) before transferring them to the GPU. Enabling this option can reduce data transfer overhead and improve training efficiency, especially for large input subtomograms when I/O or communication is a bottleneck.

To train OPUS-ET using distributed data parallel in pytorch, you can use the command,

```
torchrun --nproc_per_node=4 -m cryodrgn.commands.train_tomo_dist ribotm.star --poses ribotm_pose_euler.pkl -n 40 -b 12 --zdim 12 --lr 4.5e-5 --num-gpus 4 --multigpu --beta-control 0.5 -o . -r ../zribotmt/mask.mrc --split deep.pkl --lamb 0.5 --bfactor 3.75 --valfrac 0.1 --templateres 128 --tmp-prefix tmp --datadir /work/jpma/luo/tomo/DEF/metadata/warp_tiltseries/ --angpix 3.37 --downfrac 1. --warp --tilt-range 50 --tilt-step 2 --ctfalpha 0. --ctfbeta 1. --estpose
```

which invokes 4 processes on a 4gpu cluster, and might achieve faster training speed compared with data parallel.

The ```--encode-mode``` argument controls how latent codes are produced. ```grad``` (default) uses the encoder to produce per-particle latents and backpropagates through the encoder. ```fixed``` uses a single fixed latent code shared across particles and trains only the decoder.

Example for fixed mode training:

```
torchrun --nproc_per_node=4 -m cryodrgn.commands.train_tomo_dist pre18.star --poses pre18_pose_euler.pkl -n 120 -b 4 --zdim 12 --lr 3.e-5 --num-gpus 4 --multigpu --beta-control 0.5 -o . -r ../zribo_test/mask.mrc --split deep_splt.pkl --lamb 0.5 --bfactor 4. --valfrac 0. --templateres 160 --tmp-prefix tmp --datadir /work/jpma/luo/tomo/warp_DEF/metadata/warp_tiltseries/ --angpix 3.37 --downfrac 1. --warp --ctfalpha 0 --ctfbeta 1 --encode-mode fixed --checkpoint 10
```

If you have installed horovod according to tutorial https://github.com/alncat/opusTomo/wiki/horovod-installation, you can train OPUS-ET using

```
horovodrun -np 4 dsd train_tomo_hvd /work/ribo.star --poses ./ribo_pose_euler.pkl -n 40 -b 12 --zdim 12 --lr 4.e-5 --num-gpus 1 --multigpu --beta-control 0.5 -o /work/ribo -r ./mask.mrc --downfrac 0.9 --valfrac 0.1 --lamb 0.5 --split ribo-split.pkl --bfactor 3.5 --templateres 128 --angpix 3.37 --estpose --tmp-prefix ref --datadir /work/ --warp --tilt-range 50 --tilt-step 2 --ctfalpha 0. --ctfbeta 1. 
```
The above command will spawn 4 processes to train OPUS-ET. Using horovod can greatly reduce the overhead of IO compared to data parallel in pytorch.

The argument following train_tomo specifies the starfile for subtomograms. In contrast to OPUS-DSD, we no longer need to specify ctf since they are read from the subtomogram starfile.
Moreover, OPUS-ET needs to specify the angpix of the subtomogram by ```--angpix```, and also the prefix directory before the filename for subtomogram in starfile by ```--datadir```.

The functionality of each argument is explained in the table:
| argument |  explanation |
| --- | --- |
| --poses | pose parameters of the subtomograms in starfile |
| -n     | the number of training epoches, each training epoch loops through all subtomograms in the training set |
| -b     | the number of subtomograms for each batch on each gpu, depends on the size of available gpu memory|
| --zdim  | the dimension of composition latent encodings, increase the zdim will improve the fitting capacity of neural network, but may risk of overfitting |
| --zaffinedim | the dimension of conformation latent encodings, default is 4 |
| --lr    | the initial learning rate for adam optimizer, 5.e-5 should work fine. |
| --num-gpus | the number of gpus used for training, note that the total number of subtomograms in the total batch will be n*num-gpus |
| --multigpu |toggle on the data parallel version |
| --beta-control |the restraint strength of the beta-vae prior, the larger the argument, the stronger the restraint. The scale of beta-control should be propotional to the SNR of dataset. Suitable beta-control might help disentanglement by increasing the magnitude of latent encodings and the sparsity of latent encodings, for more details, check out [beta vae paper](https://openreview.net/forum?id=Sy2fzU9gl). In our implementation, we adjust the scale of beta-control automatically based on SNR estimation, possible ranges of this argument are [0.5-4.]. You can use larger beta-control for dataset with higher SNR|
| --beta |the schedule for restraint stengths, ```cos``` implements the [cyclic annealing schedule](https://www.microsoft.com/en-us/research/blog/less-pain-more-gain-a-simple-method-for-vae-training-with-less-of-that-kl-vanishing-agony/) and is the default option|
| -o | the directory name for storing results, such as model weights, latent encodings |
| -r | ***the solvent mask created from consensus model***, our program will focus on fitting the contents inside the mask. Since the majority part of subtomogram doesn't contain electron density, using the original subtomogram size is wasteful, by specifying a mask, our program will automatically determine a suitable crop rate to keep only the region with densities. |
| --downfrac | the downsampling fraction of input subtomogram, the input to network will be downsampled to size of D\*downfrac, where D is the original size of subtomogram. You can set it according to resolution of consensus model and the ***templateres*** you set. If you are using warp, the subtomogram is often exported as desired size, and this value can be set to 1. |
| --lamb | the restraint strength of structural disentanglement prior proposed in OPUS-DSD, set it according to the SNR of your dataset, for dataset with high SNR such as ribosome, splicesome, you can set it to 1., for dataset with lower SNR, consider lowering it. Possible ranges are [0.1, 2.]. If you find **the UMAP of embeedings is exaggeratedly stretched into a ribbon**, then the lamb you used during training is too high! |
| --split | the filename for storing the train-validation split of subtomograms |
| --valfrac | the fraction of subtomograms in the validation set, default is 0.1 |
| --bfactor | will apply exp(-bfactor/4 * s^2 * 4*pi^2/ angpix**2 ) decaying to the FT of reconstruction, s is the magnitude of frequency, increase it leads to sharper reconstruction, but takes longer to reveal the part of model with weak density since it actually dampens learning rate, possible ranges are [3, 5]. You can increase bfactor for subtomograms with larger angpix. Besides, consider using higher values for more dynamic structures.  |
| --templateres | the size of output volume of our convolutional network, it will be further resampled by spatial transformer before projecting to subtomograms. The default value is 160. You may keep it around ```D*downfrac/0.75```, which is larger than the input size. This corresponds to downsampling from the output volume of our network. You can tweak it to other resolutions, larger resolutions can generate sharper density maps, ***choices are Nx16, where N is integer between 8 and 16*** |
| --encoderres | the cubic size used to resample intermediate encoder activations before later conv layers (default 12); larger values increase memory/compute |
| --encode-mode | how latents are produced; ```grad``` trains the encoder for per-particle latents, ```fixed``` uses a shared fixed latent code and trains only the decoder |
| --plot | you can also specify this argument if you want to monitor how the reconstruction progress, our program will display the Z-projection of subtomograms and experimental subtomograms after 8 times logging intervals. Namely, you switch to interative mode by including this. The interative mode should be run using command ```python -m cryodrgn.commands.train_tomo```|
| --tmp-prefix | the prefix of intermediate reconstructions, default value is ```tmp```. OPUS-ET will output temporary reconstructions to the root directory of this program when training, whose names are ```$tmp-prefix.mrc``` |
| --angpix | the angstrom per pixel for the input subtomogram |
| --datadir | the root directory before the filename of subtomogram in input starfile |
| --estpose | estimate a pose correction for each subtomogram during training |
| --masks | the mask parameters for defining the rigid-body dynamics |
| --ctfalpha | the degree of ctf correction on the experimental subtomogram, the default value is 0, which is equivalent to phase flipping. You can also try value like 0.5, which will further correct the amplitude of FT of subtomogram by the square root of the ctf function|
| --ctfbeta | the degree of ctf correction on the reconstruction output by decoder, the default value is 1. You can also try value like 0.5, which will make the voxel values output by decoder smaller since the ctf correction is weaker.|
| --tilt-step | the interval between successive tilts, default is 2, you need to change it according to your experimental settings when training subtomograms from WARP|
| --tilt-range | the range of tilt angles, default is 50, change it to your range when training subtomograms from WARP|
| --tilt-limit | optionally limit the tilt angles used for training (e.g., restrict to a smaller angular range) |
| --warp | include this argument if the subtomograms are generated by WARP|
| --accum-step | the gradient accumulation step, default value is 1. If you set it to be n, the gradient will be accumulated over n steps before updating parameters.|
| --float16 | load subtomogram in float16, which might reduce the io overhead when distributing subtomograms to different GPUs.|

Using the plot mode will ouput the following images in the directory where you issued the training command, the fllowing images are for ATP synthase dimer:

<img width="441" height="331" alt="image" src="https://github.com/user-attachments/assets/e19105e7-34fa-409b-8cbc-2abf2e870e87" />

Each row shows a selected subtomogram and its reconstruction from a batch.
In the first row, the first image is the projection of experimental subtomogram supplemented to encoder, the second image is a 2D projection from subtomogram reconstruction blurred by the corresponding CTF, the third image is the projection of correpsonding experimental subtomogram after 3D masking.


You can use ```nohup``` to let the above command execute in background and use redirections like ```1>log 2>err``` to redirect ouput and error messages to the corresponding files.
Happy Training! **Open an issue when running into any troubles.**

To restart execution from a checkpoint, you can use
```
dsd train_tomo /work/ribo.star --poses ./ribo_pose_euler.pkl -n 40 -b 12 --zdim 12 --lr 4.e-5 --num-gpus 4 --multigpu --beta-control 0.5 -o /work/ribo -r ./mask.mrc --downfrac 0.9 --valfrac 0.1 --lamb 0.5 --split ribo-split.pkl --bfactor 3.5 --templateres 128 --angpix 3.37 --estpose --tmp-prefix ref --datadir /work/ --warp --tilt-range 50 --tilt-step 2 --ctfalpha 0. --ctfbeta 1. --load weights.9.pkl --latents z.9.pkl
```
,

```
torchrun --nproc_per_node=4 -m cryodrgn.commands.train_tomo_dist ribotm.star --poses ribotm_pose_euler.pkl -n 40 -b 12 --zdim 12 --lr 4.5e-5 --num-gpus 4 --multigpu --beta-control 0.5 -o . -r ../zribotmt/mask.mrc --split deep.pkl --lamb 0.5 --bfactor 4. --valfrac 0.1 --templateres 128 --tmp-prefix tmp --datadir /work/jpma/luo/tomo/DEF/metadata/warp_tiltseries/ --angpix 3.37 --downfrac 1. --warp --tilt-range 50 --tilt-step 2 --ctfalpha 0. --ctfbeta 1. --estpose --load weights.9.pkl --latents z.9.pkl
```

or

```
horovodrun -np 4 dsd train_tomo_hvd /work/ribo.star --poses ./ribo_pose_euler.pkl -n 40 -b 12 --zdim 12 --lr 4.e-5 --num-gpus 1 --multigpu --beta-control 0.5 -o /work/ribo -r ./mask.mrc --downfrac 0.9 --valfrac 0.1 --lamb 0.5 --split ribo-split.pkl --bfactor 4. --templateres 128 --angpix 3.37 --estpose --tmp-prefix ref --datadir /work/ --warp --tilt-range 50 --tilt-step 2 --ctfalpha 0. --ctfbeta 1. --load weights.9.pkl --latents z.9.pkl
```

| argument |  explanation |
| --- | --- |
| --load | the weight checkpoint from the restarting epoch |
| --latents | the latent encodings from the restarting epoch |

both are in the output directory

During training, OPUS-ET will output temporary volumes called ```tmp*.mrc``` (or the prefix you specified), you can check the intermediate results by viewing them in Chimera. OPUS-ET reads subotomograms from disk as needed during training.

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
The first step is to sample the latent space using kmeans and PCA algorithms. Suppose the training results are in ```/work/ribo```, 
```
dsdsh analyze /work/ribo 16 12 20
                $1    $2 $3 $4
```

- $1 is the output directory used in training, which stores ```weights.*.pkl, z.*.pkl, config.pkl```
- $2 is the epoch number you would like to analyze,
- $3 is the number of PCs you would like to sample for traversal
- $4 is the number of clusters for kmeans clustering.

The analysis result will be stored in ```/work/ribo/analyze.16``` and ```/work/ribo/defanalyze.16```, i.e., the output directory plus the epoch number you analyzed, using the above command, and ```defanalyze``` stands for the anlysis result for dynamics latent space. You can find the UMAP with the labeled kmeans centers in /work/ribo/analyze.16/kmeans16/umap.png and the umap with particles colored by their projection parameter in /work/ribo/analyze.16/umap.png .
The sampled latent codes are in ```analyze.16/kmeans20/centers.txt``` for kmeans sampling, and ```analyze.16/pcN/z_pc.txt``` for PCA trajectories in composition latent space.
The sampled latent codes for conformation latent space are stored in ```defanalyze.16``` in similar conventiion.

After executing the analysis once, you may skip the lengthy umap embedding laterly by appending ```--skip-umap``` to the command in analyze.sh. Our analysis script will read the pickled umap embeddings directly.
Moreover, you can perform a second round of analysis on selected clusters from the first round, e.g.,
```
dsdsh analyze /work/ribo 16 12 30 --kpc 20:2-5,8
```
The ```--kpc``` argument specifies that the analysis is based on the KMeans clustering result ```analyze.16/kmeans20```, using classes ```2 to 5, and 8```.
The grammar of ```--kpc``` is that the first value before ```:``` is the number of clusters used in previous analysis, ```x-y``` means using classes from x to y, 
and ```,``` is used as separation. This command will create a new folder ```analyze.filter.16```, which stores the analysis result.
This kinds of hierarhical clustering could be very effective at identifying sparsely populated species, or reconstruting intra-class variations along PCs by focusing on several clusters.

Finally, you may ask OPUS-ET to output a joint latent code, which combine composition and conformation latent codes by specifying ```--joint```, i.e.,
```
dsdsh analyze /work/ribo 16 12 30 --kpc 20:2-5,8 --joint
```
will output ```analyze.filter.16/kmeans30/centers_joint.txt``` additionally, which concatanate the corresponding conformation latent codes to composition cluster 
centroids and allows you to reconstruct conformations with full latent space laterly.

## reconstruct volumes <div id="reconstruct">
Next, you can reconstruct volumes at those latent codes sampled by KMeans and PCA.

The ```dsdsh eval_vol``` command has following options:
You can generate the volume which corresponds to KMeans cluster centroid using

```
dsdsh eval_vol /work/sp 16 kmeans 16 2.2
                 $1     $2   $3   $4  $5
```

- $1 is the output directory used in training, which stores ```weights.*.pkl, z.*.pkl, config.pkl``` and the clustering result
- $2 is the epoch number you just analyzed, if you have performed second round analysis, you can specify ```filter.N```
- $3 specifies the kind of analysis result where volumes are generated, i.e., kmeans clusters or principal components, use ```kmeans``` for kmeans clustering, ```pc```
for principal components, ```dpc``` for conformation principal components, ```joint``` for kmeans clustering with corresponding conformation latent codes,
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

To reconstruct trajectories along conformation PC, you need to select a template in composition latent space, which is specified via ```--kmeans K --dfk N```, e.g.,
```
 dsdsh eval_vol . 16 dpc 1 2.2 --kmeans 16 --dfk 2 --masks ../mask2new.pkl
```
generates volumes along dpc1, i.e., the first principal component of dynamics latent space, and use the class 2 in kmeans16 for epoch 16 as template. 
You can find volumes in ```/work/sp/defanalyze.16/pc1```.

## select particles <div id="select">
Finally, you can also retrieve the star files for subtomograms in each kmeans cluster using

```
dsdsh parse_pose /work/consensus_data.star 320 1.699 /work/sp 16 20 --relion31
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
With those starfiles, you can easily obtain the desired population of biomolecules by combining clusters with expected reconstructions.

Equivalently, you can use the command
```
dsd parse_pose_star /work/consensus_data.star -D 128 --Apix 3.37 --labels analyze.filter.19/kmeans20/labels.pkl --outdir analyze.filter.19/kmeans20/
```
to split the ```consensus_data.star``` into different subsets for different clusters.

Lastly, you can obtain the reconstruction using OPUS-ET by training the subset with ```--encode-mode fixed```.

## Interactive filtering with cryoDRGN_filtering_template.ipynb <div id="filtering">

For interactive particle selection and filtering, OPUS-ET provides a Jupyter notebook template at `cryodrgn/templates/cryoDRGN_filtering_template.ipynb`. This notebook enables visual exploration of the latent space and selection of particles using multiple interactive methods:

- **Clustering-based selection**: Select particles belonging to specific k-means or Gaussian Mixture Model (GMM) clusters.
- **Outlier detection**: Identify particles with latent vector magnitudes beyond a specified Z-score threshold.
- **Lasso selection**: Manually select particles by drawing regions on UMAP or PCA plots.
- **Interactive visualization**: Explore latent space embeddings with plotly-powered interactive plots.

### Usage

1. Perform analysis, the notebook template will be copied to your analysis directory:
   ```
   dsdsh analyze . 39 10 20
   ```

2. Open the notebook with Jupyter and configure the `WORKDIR` and `EPOCH` variables to point to your training output directory.

3. Run the cells sequentially to:
   - Load latent encodings and pose parameters
   - Visualize latent space (PCA, UMAP, pose distributions)
   - Apply interactive filtering methods
   - Export selected particles as STAR files

4. The notebook tracks selected particles in the variable `ind_selected`. Selected particles can be saved as a `.star` file for downstream processing in RELION or other tools.

**Requirements:** The notebook requires `jupyter-notebook`, `anywidget`, `plotly`, and `ipywidgets` for interactive visualizations. Install them via:
```bash
pip install notebook plotly anywidget ipywidgets
```

**Tip:** The interactive filtering notebook is particularly useful for cleaning template matching results, removing outliers, and selecting homogeneous particle subsets for high-resolution refinement.


## OPUS-ET Analysis Skill for Kimi Code CLI <div id="skill">

OPUS-ET includes a skill file `opus-et-analysis.skill` that provides AI-assisted analysis workflows when using Kimi Code CLI. This skill encapsulates best practices for processing training results and can guide you through complex analysis pipelines including PCA/k-means clustering, volume generation, pose parsing, and STAR file manipulation.

### Installing the Skill

To use the skill in Kimi Code CLI:

```bash
# Import the skill into Kimi Code CLI
kimi skill import opus-et-analysis.skill

# Or copy to the skills directory manually
cp opus-et-analysis.skill ~/.kimi/skills/
```

### Quick Start with the Skill

Once installed, you can ask Kimi Code CLI to help with analysis tasks:

```
"Analyze epoch 39 with 10 PCs and 20 clusters"
"Generate volumes for all k-means centers from epoch 39"
"Combine clusters 9, 10, 11, 12 and create a pose pickle"
"Run deformation analysis along PC1 using cluster 17 as template"
```

The skill will automatically extract parameters from `config.pkl` and generate the appropriate commands.

### Key Configuration Parameters

When working with OPUS-ET results, these key parameters are extracted from `config.pkl`:

| Parameter | Source | Description |
|-----------|--------|-------------|
| `Apix` | `config['model_args']['Apix']` | Angstrom per pixel |
| `D` | `config['lattice_args']['D'] - 1` | **Effective box size (lattice D minus 1)** |
| `particles` | `config['dataset_args']['particles']` | Original STAR file path |
| `zdim` | `config['model_args']['zdim']` | Latent dimension |

**Important:** The effective box size for `parse_pose_star` is always `lattice_args['D'] - 1`, not the raw D value.

Use the included helper script to extract these values:
```bash
python opus-et-analysis/scripts/extract_config.py config.pkl
```

### Standard Analysis Workflows

#### 1. Analyze Epoch (PCA + K-means)

Run PCA and k-means clustering on a specific epoch:

```bash
dsdsh analyze <workdir> <epoch> <numpc> <numk>
```

Example:
```bash
dsdsh analyze . 39 10 20
```

Output: `analyze.39/` directory with `kmeans20/`, `pc1/` to `pc10/`, and visualization plots.

#### 2. Generate Volumes for K-means Centers

```bash
dsd eval_vol --load weights.39.pkl \
    -c config.pkl \
    -o kmeans_volumes \
    --zfile analyze.39/kmeans20/centers.txt \
    --Apix 3.37 \
    --prefix kmeans_cluster
```

#### 3. Generate Volumes for Principal Components

```bash
mkdir -p pc_volumes/pc1
dsd eval_vol --load weights.39.pkl \
    -c config.pkl \
    -o pc_volumes/pc1 \
    --zfile analyze.39/pc1/z_pc.txt \
    --Apix 3.37 \
    --prefix pc1
```

#### 4. Create Star Files for Clusters

Parse poses and split by k-means cluster labels:

```bash
# Extract config to get correct D value
python opus-et-analysis/scripts/extract_config.py config.pkl
# Output: Apix=3.37, D=127 (effective box size)

# Parse with correct box size (D-1 from lattice_args)
dsd parse_pose_star ribotm.star \
    -D 127 \
    --Apix 3.37 \
    --labels analyze.39/kmeans20/labels.pkl \
    --outdir kmeans_pose
```

#### 5. Combine Star Files

Merge multiple cluster STAR files:

```bash
# Chain combine commands for multiple files
dsdsh combine_star pre9.star pre10.star temp1.star
dsdsh combine_star temp1.star pre11.star temp2.star
dsdsh combine_star temp2.star pre12.star combined_9_10_11_12.star
```

#### 6. Generate Pose Pickle for Combined Clusters

```bash
dsd parse_pose_star kmeans_pose/combined_9_10_11_12.star \
    -D 127 \
    --Apix 3.37 \
    -o kmeans_pose/combined_9_10_11_12_pose.pkl
```

### Deformation Analysis Workflow

For models trained with `--masks` (rigid body motion), the analysis generates **both** conformation and deformation latent spaces:

```bash
# Single command generates both analyze.39/ and defanalyze.39/
dsdsh analyze . 39 10 30
```

| Directory | Content | Dimensions | Use Case |
|-----------|---------|------------|----------|
| `analyze.<epoch>/` | Full composition latent space | Model zdim (e.g., 12) | Standard volume generation |
| `defanalyze.<epoch>/` | Deformation parameter space | Conformational zdim (e.g., 4 for 2-body) | Rigid body motion analysis |

#### Generate Deformation Volumes Along PCs

```bash
# Step 1: Extract k-means center as template
python3 << 'PYEOF'
import pickle
import numpy as np
centers = pickle.load(open('analyze.39/kmeans30/centers.pkl', 'rb'))
np.savetxt('template_z17.txt', centers[17].reshape(1, -1), fmt='%.6f')
PYEOF

# Step 2: Generate deformation volumes
dsd eval_vol --load weights.39.pkl \
    -c config.pkl \
    -o defanalyze_volumes/pc1 \
    --deform \
    --masks mask_params.pkl \
    --template-z template_z17.txt \
    --template-z-ind 0 \
    --zfile defanalyze.39/pc1/z_pc.txt \
    --Apix 3.37 \
    --prefix reference
```

**Key parameters for deformation mode:**
- `--deform`: Enable deformation mode
- `--masks`: Path to `mask_params.pkl` with rigid body definitions
- `--template-z`: Base conformation z-values (2D: rows × zdim)
- `--template-z-ind`: Template row index (0 for first row)
- `--zfile`: Deformation parameters from `defanalyze.<epoch>/`

#### Inspect Mask Parameters

```python
import torch
m = torch.load('mask_params.pkl', map_location='cpu')
print('Number of bodies:', m['com_bodies'].shape[0])
print('COM of bodies:', m['com_bodies'])
print('Principal axes:', m['principal_axes'])
```

### Complete Workflow Example

Full pipeline from training results to combined particle selection:

```bash
# 1. Analyze epoch
dsdsh analyze . 39 10 20

# 2. Generate volumes for k-means centers
dsd eval_vol --load weights.39.pkl -c config.pkl -o kmeans_volumes \
    --zfile analyze.39/kmeans20/centers.txt --Apix 3.37 --prefix kmeans_cluster

# 3. Create star files for all clusters
dsd parse_pose_star ribotm.star -D 127 --Apix 3.37 \
    --labels analyze.39/kmeans20/labels.pkl --outdir kmeans_pose

# 4. Combine specific clusters
dsdsh combine_star kmeans_pose/pre9.star kmeans_pose/pre10.star temp.star
dsdsh combine_star temp.star kmeans_pose/pre11.star temp2.star
dsdsh combine_star temp2.star kmeans_pose/pre12.star \
    kmeans_pose/combined_9_10_11_12.star

# 5. Generate pose pickle for combined clusters
dsd parse_pose_star kmeans_pose/combined_9_10_11_12.star \
    -D 127 --Apix 3.37 -o kmeans_pose/combined_9_10_11_12_pose.pkl
```

### Directory Structure Convention

Standard output structure after analysis:

```
.
├── analyze.<epoch>/              # Conformation latent space analysis
│   ├── kmeans<numk>/
│   │   ├── centers.txt          # Cluster center latent codes
│   │   ├── centers.pkl          # Numpy array of centers
│   │   ├── labels.pkl           # Particle cluster assignments
│   │   ├── centers_ind.txt      # Particle indices closest to centers
│   │   └── pre<N>.star          # Star files per cluster
│   ├── pc<N>/
│   │   └── z_pc.txt             # PC trajectory latent codes
│   └── *.png                    # UMAP, PCA plots
├── defanalyze.<epoch>/           # Deformation analysis (if --masks used)
│   └── similar structure to analyze.<epoch>/
├── kmeans_volumes/               # Generated cluster volumes
├── pc_volumes/                   # Generated PC trajectory volumes
└── kmeans_pose/                  # Split star files by cluster
```

For complete command reference and advanced workflows, refer to the skill documentation at `opus-et-analysis/references/commands.md` or ask Kimi Code CLI: `"Show me OPUS-ET analysis workflows"`.

---

This program is built upon a set of great works:
- [opusDSD](https://github.com/alncat/opusDSD)
- [cryoDRGN](https://github.com/zhonge/cryodrgn)
- [Neural Volumes](https://stephenlombardi.github.io/projects/neuralvolumes/)
- [UMAP](https://umap-learn.readthedocs.io/en/latest/)
- [Healpy](https://healpy.readthedocs.io/en/latest/)
