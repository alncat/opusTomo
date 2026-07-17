# mypackage/myscript_wrapper.py
import subprocess
import os

class eval_vol:
    @classmethod
    def add_args(cls, parser):
        parser.add_argument('resdir', type=os.path.abspath, help='result directory')
        parser.add_argument('N', type=str, help='epoch number')
        parser.add_argument('method', choices=('kmeans','pc', 'dpc', 'joint'), default='kmeans', help='which latent codes to reconstruct (default: %(default)s): '
                            'kmeans=KMeans cluster centers, pc=a PC traversal, joint=KMeans centers with concatenated conformation codes (centers_joint.txt), '
                            'dpc=multi-body deformation dynamics along a PC')
        parser.add_argument('num', type=int, help='the number of KMeans clusters or PCs for reconstruction')
        parser.add_argument('apix', type=float, help='desired apix of the output volume')
        parser.add_argument('--num-bodies', default=0, type=int, required=False, help='the number of bodies defined in training (default: %(default)s)')
        parser.add_argument('--masks', type=os.path.abspath, required=False, help='path to the pkl for masks params')
        parser.add_argument('--kmeans', type=int, required=False, help='the kmeans folder to select the template to be deformed')
        parser.add_argument('--dfk', type=int, required=False, help='the kmeans center serving as the template to be deformed')
        parser.add_argument('--flip', action='store_true', required=False, help='invert handness of the reconstruction')

    @classmethod
    def main(cls, args):
        script_path = os.path.join(os.path.dirname(__file__), 'eval_vol.sh')
        if args.flip:
            flip='--flip'
        else:
            flip=''
        if args.method == 'dpc':
            assert args.kmeans is not None and args.dfk is not None
            subprocess.call(['bash', script_path, args.resdir, str(args.N), args.method, str(args.num), str(args.apix), args.masks, str(args.kmeans), str(args.dfk), flip])
        else:
            subprocess.call(['bash', script_path, args.resdir, str(args.N), args.method, str(args.num), str(args.apix), str(args.num_bodies), flip])

class analyze:
    @classmethod
    def add_args(cls, parser):
        parser.add_argument('resdir', type=os.path.abspath, help='result directory')
        parser.add_argument('N', type=int, help='epoch number to be analyzed')
        parser.add_argument('numpc', type=int, help='number of PCs')
        parser.add_argument('numk', type=int, help='number of KMeans clusters')
        parser.add_argument('--skip-umap', action='store_true', required=False, help='instead of learn a umap embedding, loading one from umap.pkl (default: %(default)s)')
        parser.add_argument('--kpc', type=str, required=False, help='re-cluster a subset of a previous KMeans run, as K:classes, e.g. 20:2-5,8 (K is the previous cluster count; x-y is an inclusive class range) (default: %(default)s)')
        parser.add_argument('--joint', action='store_true', required=False, help='also write centers_joint.txt: composition centers with the matching conformation latents concatenated (default: %(default)s)')
    @classmethod
    def main(cls, args):
        script_path = os.path.join(os.path.dirname(__file__), 'analyze.sh')
        if args.kpc:
            kpc = '--kpc ' + str(args.kpc)
        else:
            kpc = ''
        if args.joint:
            joint = '--joint'
        else:
            joint = ''
        if args.skip_umap:
            subprocess.call(['bash', script_path, args.resdir, str(args.N), str(args.numpc), str(args.numk), '--skip-umap', kpc, joint])
        else:
            subprocess.call(['bash', script_path, args.resdir, str(args.N), str(args.numpc), str(args.numk), kpc, joint])

class parse_pose:
    @classmethod
    def add_args(cls, parser):
        parser.add_argument('starfile', type=os.path.abspath, help='starfile for images')
        parser.add_argument('D', type=int, help='the size of image in the input stack')
        parser.add_argument('apix', type=float, help='the apix of the input stack')
        parser.add_argument('resdir', type=os.path.abspath, help='result folder storing training results')
        parser.add_argument('N', type=int, help='epoch number')
        parser.add_argument('kmeans', type=int, help='KMeans clusters for classification')
        parser.add_argument('--relion31', action='store_true', help='if the input starfile is of version 3.1')

    @classmethod
    def main(cls, args):
        script_path = os.path.join(os.path.dirname(__file__), 'parse_pose.sh')
        if args.relion31:
            subprocess.call(['bash', script_path, args.starfile, str(args.D), str(args.apix), args.resdir, str(args.N), str(args.kmeans), '--relion31'])
        else:
            subprocess.call(['bash', script_path, args.starfile, str(args.D), str(args.apix), args.resdir, str(args.N), str(args.kmeans),])

class combine_star:
    @classmethod
    def add_args(cls, parser):
        parser.add_argument('starfile1', type=os.path.abspath, help='starfile for first set')
        parser.add_argument('starfile2', type=os.path.abspath, help='starfile for second set')
        parser.add_argument('starfileout', type=os.path.abspath, help='filename for the output')

    @classmethod
    def main(cls, args):
        script_path = os.path.join(os.path.dirname(__file__), 'union_star.py')
        subprocess.call(['python', script_path, args.starfile1, args.starfile2, args.starfileout])

class convert_warp:
    @classmethod
    def add_args(cls, parser):
        parser.add_argument('starfile', type=os.path.abspath, help='starfile in WARP format')
        parser.add_argument('angpix', type=float, help='angstrom per pixel of the tilt series')
        parser.add_argument('--kv', default=300, type=float, help='voltage of tilt series, (default: %(default)s)')
        parser.add_argument('--cs', default=2.7, type=float, help='spherical aberration of tilt series, (default: %(default)s)')

    @classmethod
    def main(cls, args):
        script_path = os.path.join(os.path.dirname(__file__), 'convert_warp.py')
        subprocess.call(['python', script_path, args.starfile, str(args.angpix), str(args.kv), str(args.cs)])

class convert_pytom:
    @classmethod
    def add_args(cls, parser):
        parser.add_argument('starfile', type=os.path.abspath, help='starfile in WARP format')
        parser.add_argument('mic', type=str, help='the name of selected micrograph')

    @classmethod
    def main(cls, args):
        script_path = os.path.join(os.path.dirname(__file__), 'convert_pytom.py')
        subprocess.call(['python', script_path, args.starfile, str(args.mic)])

class convert_pytom_to_star:
    @classmethod
    def add_args(cls, parser):
        parser.add_argument('--input', type=os.path.abspath, required=False, help='single input XML file')
        parser.add_argument('--input-dir', type=os.path.abspath, required=False, help='directory containing XML files')
        parser.add_argument('--pattern', type=str, default='*.xml', required=False,
                            help='file pattern for batch conversion (default: %(default)s)')
        parser.add_argument('--output', type=os.path.abspath, required=True, help='output STAR file path')
        parser.add_argument('--pixel-size', type=float, default=7.84, required=False,
                            help='detector pixel size in Angstroms (default: %(default)s)')
        parser.add_argument('--bin-size', type=float, default=4, required=False,
                            help='binning factor used in PyTom (default: %(default)s)')
        parser.add_argument('--magnification', type=float, default=10000, required=False,
                            help='magnification (default: %(default)s)')
        parser.add_argument('--ctf-template', type=str, default=None, required=False,
                            help='CTF image path template with {ts_name} or {i}')
        parser.add_argument('--workers', type=int, default=10, required=False,
                            help='number of workers for batch conversion (default: %(default)s)')

    @classmethod
    def main(cls, args):
        script_path = os.path.join(os.path.dirname(__file__), 'convert_pytom_to_star.py')
        cmd = ['python', script_path, '--output', args.output,
               '--pattern', args.pattern,
               '--pixel-size', str(args.pixel_size),
               '--bin-size', str(args.bin_size),
               '--magnification', str(args.magnification),
               '--workers', str(args.workers)]
        if args.input is not None:
            cmd.extend(['--input', args.input])
        if args.input_dir is not None:
            cmd.extend(['--input-dir', args.input_dir])
        if args.ctf_template is not None:
            cmd.extend(['--ctf-template', args.ctf_template])
        subprocess.call(cmd)

class convert_artiax:
    @classmethod
    def add_args(cls, parser):
        parser.add_argument('input', type=os.path.abspath, help='input STAR file')
        parser.add_argument('mic', type=str, help='tomogram/micrograph id')
        parser.add_argument('--factor', type=float, default=2.132, required=False,
                            help='coordinate scaling factor (default: %(default)s)')
        parser.add_argument('--deduplicate', action='store_true', required=False,
                            help='remove duplicates by rlnImageName')
        parser.add_argument('--micrograph-col', type=str, default='rlnMicrographName', required=False,
                            help='micrograph column name (default: %(default)s)')
        parser.add_argument('--data-block', type=str, required=False,
                            help='STAR data block name for multi-block STAR')
        parser.add_argument('-o', '--output', type=os.path.abspath, required=False,
                            help='output STAR path (default: <input_stem>_<mic>.star)')

    @classmethod
    def main(cls, args):
        script_path = os.path.join(os.path.dirname(__file__), 'convert_artiax.py')
        cmd = ['python', script_path, args.input, args.mic,
               '--factor', str(args.factor),
               '--micrograph-col', args.micrograph_col]
        if args.deduplicate:
            cmd.append('--deduplicate')
        if args.data_block is not None:
            cmd.extend(['--data-block', args.data_block])
        if args.output is not None:
            cmd.extend(['--output', args.output])
        subprocess.call(cmd)

class convert_star:
    @classmethod
    def add_args(cls, parser):
        parser.add_argument('starfile', type=os.path.abspath, help='input starfile')
        parser.add_argument('angpix', type=float, help='angstrom per pixel of the tilt series')
        parser.add_argument('--rescale-angpix', type=float, required=False,
                            help='target angpix to rescale pixel coordinates/translations')
        parser.add_argument('--subset-label', type=int, required=False,
                            help='when set, also write out only this _rlnRandomSubset label')
        parser.add_argument('--remove-symexp', action='store_true', required=False,
                            help='remove symmetry expansion using one row per rlnImageName')

    @classmethod
    def main(cls, args):
        script_path = os.path.join(os.path.dirname(__file__), 'convert_star.py')
        cmd = ['python', script_path, args.starfile, str(args.angpix)]
        if args.rescale_angpix is not None:
            cmd.extend(['--rescale-angpix', str(args.rescale_angpix)])
        if args.subset_label is not None:
            cmd.extend(['--subset-label', str(args.subset_label)])
        if args.remove_symexp:
            cmd.append('--remove-symexp')
        subprocess.call(cmd)

class extract_tomo_cubes:
    @classmethod
    def add_args(cls, parser):
        parser.add_argument('tomo', type=os.path.abspath, help='input tomogram (.mrc/.map)')
        parser.add_argument('star', type=os.path.abspath, help='input STAR with coordinates')
        parser.add_argument('--coord-cols', nargs=3, metavar=('XCOL', 'YCOL', 'ZCOL'),
                            default=('rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ'),
                            help='coordinate columns in STAR')
        parser.add_argument('--data-block', type=str, required=False,
                            help='STAR data block name when STAR has multiple blocks')
        parser.add_argument('--tomo-col', type=str, required=False,
                            help='STAR column used to select one tomogram')
        parser.add_argument('--tomo-id', type=str, required=False,
                            help='value in --tomo-col for selected tomogram')
        parser.add_argument('--box-size', type=int, required=True, help='cube size in voxels')
        parser.add_argument('--coord-scale', type=float, default=1.0, required=False,
                            help='scale factor for STAR coordinates')
        parser.add_argument('--one-based', action='store_true', required=False,
                            help='treat coordinates as 1-based')
        parser.add_argument('--round-mode', choices=('round', 'floor', 'ceil'), default='round',
                            required=False, help='rounding mode for float coordinates')
        parser.add_argument('--pad-outside', action='store_true', required=False,
                            help='pad cubes crossing boundary instead of skipping')
        parser.add_argument('--out-star', type=os.path.abspath, required=False,
                            help='optional STAR with only extracted entries')
        parser.add_argument('-o', '--output-dir', type=os.path.abspath, required=True,
                            help='output directory for extracted cubes')
        parser.add_argument('--prefix', type=str, default='cube', required=False,
                            help='prefix for output cube filenames')
        parser.add_argument('--start-index', type=int, default=0, required=False,
                            help='starting index for output filenames')
        parser.add_argument('--write-stack', type=os.path.abspath, required=False,
                            help='optional stack output path (.mrcs/.mrc)')

    @classmethod
    def main(cls, args):
        script_path = os.path.join(os.path.dirname(__file__), 'extract_tomo_cubes.py')
        cmd = ['python', script_path, args.tomo, args.star,
               '--box-size', str(args.box_size),
               '--coord-scale', str(args.coord_scale),
               '--round-mode', args.round_mode,
               '-o', args.output_dir,
               '--prefix', args.prefix,
               '--start-index', str(args.start_index)]
        if args.coord_cols is not None:
            cmd.extend(['--coord-cols', args.coord_cols[0], args.coord_cols[1], args.coord_cols[2]])
        if args.data_block is not None:
            cmd.extend(['--data-block', args.data_block])
        if args.tomo_col is not None:
            cmd.extend(['--tomo-col', args.tomo_col])
        if args.tomo_id is not None:
            cmd.extend(['--tomo-id', args.tomo_id])
        if args.one_based:
            cmd.append('--one-based')
        if args.pad_outside:
            cmd.append('--pad-outside')
        if args.out_star is not None:
            cmd.extend(['--out-star', args.out_star])
        if args.write_stack is not None:
            cmd.extend(['--write-stack', args.write_stack])
        subprocess.call(cmd)

class create_mask:
    @classmethod
    def add_args(cls, parser):
        parser.add_argument('input', type=os.path.abspath, help='input volume (.mrc/.map)')
        parser.add_argument('-o', '--output', type=os.path.abspath, required=True, help='output mask MRC')
        parser.add_argument('--threshold', type=float, required=False, help='absolute threshold for mask')
        parser.add_argument('--percentile', type=float, default=99.0, required=False,
                            help='percentile threshold if --threshold not set (default: %(default)s)')
        parser.add_argument('--use-abs', action='store_true', required=False,
                            help='threshold on absolute density')
        parser.add_argument('--largest-component', action='store_true', required=False,
                            help='keep only largest connected component')
        parser.add_argument('--fill-holes', action='store_true', required=False, help='fill holes')
        parser.add_argument('--open', type=int, default=0, required=False, help='opening iterations')
        parser.add_argument('--close', type=int, default=0, required=False, help='closing iterations')
        parser.add_argument('--dilate', type=int, default=0, required=False, help='dilation iterations')
        parser.add_argument('--erode', type=int, default=0, required=False, help='erosion iterations')
        parser.add_argument('--soft-edge', type=float, default=0.0, required=False,
                            help='soft edge width in voxels')
        parser.add_argument('--sphere-radius', type=float, default=None, required=False,
                            help='create spherical mask with this radius (voxels)')
        parser.add_argument('--sphere-center', type=float, nargs=3, required=False,
                            metavar=('CZ', 'CY', 'CX'),
                            help='sphere center in voxel coordinates (z y x)')

    @classmethod
    def main(cls, args):
        script_path = os.path.join(os.path.dirname(__file__), 'create_mask.py')
        cmd = ['python', script_path, args.input, '-o', args.output,
               '--percentile', str(args.percentile),
               '--open', str(args.open), '--close', str(args.close),
               '--dilate', str(args.dilate), '--erode', str(args.erode),
               '--soft-edge', str(args.soft_edge)]
        if args.threshold is not None:
            cmd.extend(['--threshold', str(args.threshold)])
        if args.use_abs:
            cmd.append('--use-abs')
        if args.largest_component:
            cmd.append('--largest-component')
        if args.fill_holes:
            cmd.append('--fill-holes')
        if args.sphere_radius is not None:
            cmd.extend(['--sphere-radius', str(args.sphere_radius)])
        if args.sphere_center is not None:
            cmd.extend(['--sphere-center',
                        str(args.sphere_center[0]),
                        str(args.sphere_center[1]),
                        str(args.sphere_center[2])])
        subprocess.call(cmd)

class prepare:
    @classmethod
    def add_args(cls, parser):
        parser.add_argument('starfile', type=os.path.abspath, help='starfile for images')
        parser.add_argument('D', type=int, help='the size of image in the input stack')
        parser.add_argument('apix', type=float, help='the apix of the input stack')
        parser.add_argument('--relion31', action='store_true', help='whether the input starfile is of version 3.1')

    @classmethod
    def main(cls, args):
        script_path = os.path.join(os.path.dirname(__file__), 'prepare.sh')
        if args.relion31:
            subprocess.call(['bash', script_path, args.starfile, str(args.D), str(args.apix), '--relion31'])
        else:
            subprocess.call(['bash', script_path, args.starfile, str(args.D), str(args.apix),])

class prepare_multi:
    @classmethod
    def add_args(cls, parser):
        parser.add_argument('starfile', type=os.path.abspath, help='starfile for images')
        parser.add_argument('D', type=int, help='the size of image in the input stack')
        parser.add_argument('apix', type=float, help='the apix of the input stack')
        parser.add_argument('masks', type=os.path.abspath, help='starfile storing mask definitions for multi-body refinement')
        parser.add_argument('numb', type=int, help='the number of bodies defined for multi-body refinement')
        parser.add_argument('--volumes', type=os.path.abspath, help='the path to the volume series generated from PCA for defining rotation axes')
        parser.add_argument('--relion31', action='store_true', help='whether the input starfile is of version 3.1')
        parser.add_argument('--outmasks', default='mask_params', help='the name of pkl file storing masks related parameters, \
                            you should omit the filetype name .pkl, (default: %(default)s)')

    @classmethod
    def main(cls, args):
        script_path = os.path.join(os.path.dirname(__file__), 'prepare_multi.sh')
        if args.relion31:
            if args.volumes:
                subprocess.call(['bash', script_path, args.starfile, str(args.D), str(args.apix), args.masks, str(args.numb), '--volumes ' + args.volumes, '--outmasks ' + args.outmasks, '--relion31',])
            else:
                subprocess.call(['bash', script_path, args.starfile, str(args.D), str(args.apix), args.masks, str(args.numb), '', '--outmasks ' + args.outmasks, '--relion31'])
        else:
            if args.volumes:
                subprocess.call(['bash', script_path, args.starfile, str(args.D), str(args.apix), args.masks, str(args.numb), '--volumes ' + args.volumes, '--outmasks ' + args.outmasks,])
            else:
                subprocess.call(['bash', script_path, args.starfile, str(args.D), str(args.apix), args.masks, str(args.numb), '--outmasks ' + args.outmasks,])
