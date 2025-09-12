# mypackage/__main__.py
import argparse
from . import wrapper
#from .module2 import function2

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    # Subparser for command1
    parser_command1 = subparsers.add_parser('eval_vol', help='reconstruct volumes')
    # Add arguments for command1 if needed
    wrapper.eval_vol.add_args(parser_command1)
    parser_command1.set_defaults(func=wrapper.eval_vol.main)

    # Subparser for command2
    parser_command2 = subparsers.add_parser('analyze', help='analyze latent space')
    # Add arguments for command2 if needed
    wrapper.analyze.add_args(parser_command2)
    parser_command2.set_defaults(func=wrapper.analyze.main)

    # Subparser for command2
    parser_command2 = subparsers.add_parser('parse_pose', help='split starfile according to clustering')
    # Add arguments for command2 if needed
    wrapper.parse_pose.add_args(parser_command2)
    parser_command2.set_defaults(func=wrapper.parse_pose.main)

    # Subparser for command2
    parser_command2 = subparsers.add_parser('combine_star', help='combine two starfiles')
    # Add arguments for command2 if needed
    wrapper.combine_star.add_args(parser_command2)
    parser_command2.set_defaults(func=wrapper.combine_star.main)

    # Subparser for command2
    parser_command2 = subparsers.add_parser('convert_warp', help='convert warp/m refined starfile to relion\'s format')
    # Add arguments for command2 if needed
    wrapper.convert_warp.add_args(parser_command2)
    parser_command2.set_defaults(func=wrapper.convert_warp.main)

    # Subparser for command2
    parser_command2 = subparsers.add_parser('convert_pytom', help='convert pytom\'s starfile from template matching to warp\'s format')
    # Add arguments for command2 if needed
    wrapper.convert_pytom.add_args(parser_command2)
    parser_command2.set_defaults(func=wrapper.convert_pytom.main)

    # Subparser for command2
    parser_command2 = subparsers.add_parser('convert_star', help='split starfiles by even/odd')
    # Add arguments for command2 if needed
    wrapper.convert_star.add_args(parser_command2)
    parser_command2.set_defaults(func=wrapper.convert_star.main)

    # Subparser for command2
    parser_command2 = subparsers.add_parser('prepare', help='prepare input files for training opus-tomo')
    # Add arguments for command2 if needed
    wrapper.prepare.add_args(parser_command2)
    parser_command2.set_defaults(func=wrapper.prepare.main)

    # Subparser for command2
    parser_command2 = subparsers.add_parser('prepare_multi', help='prepare input files for training opus-tomo with multibody dynamics')
    # Add arguments for command2 if needed
    wrapper.prepare_multi.add_args(parser_command2)
    parser_command2.set_defaults(func=wrapper.prepare_multi.main)

    args = parser.parse_args()
    if args.command:
        args.func(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()

