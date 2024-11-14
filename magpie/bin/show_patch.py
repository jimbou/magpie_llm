import argparse
import configparser
import pathlib

import magpie

# ================================================================================
# Main function
# ================================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Magpie show patch')
    parser.add_argument('--scenario', type=pathlib.Path, required=True)
    parser.add_argument('--patch', type=str, required=True)
    parser.add_argument('--keep', action='store_true')
    args = parser.parse_args()

    # read scenario file
    config = configparser.ConfigParser()
    config.read_dict(magpie.core.default_scenario)
    config.read(args.scenario)
    magpie.core.pre_setup(config)

    # recreate patch
    if args.patch.endswith('.patch'):
        with pathlib.Path(args.patch).open('r') as f:
            args.patch = f.read().strip()
    patch = magpie.core.Patch.from_string(args.patch)

    # setup
    magpie.core.setup(config)
    software = magpie.utils.software_from_string(config['software']['software'])(config)
    variant = magpie.core.Variant(software, patch)

    # show patch
    msg = '==== REPORT ===='
    if magpie.settings.color_output:
        msg = f'\033[1m{msg}\033[0m'
    software.logger.info(msg)
    software.logger.info('Patch: %s', patch)
    diff = variant.diff
    if magpie.settings.color_output:
        diff = magpie.core.BasicProtocol.color_diff(diff)
    software.logger.info('Diff:\n%s', diff)
    if args.keep:
        software.logger.info('Artefact: %s', software.work_dir)
        software.write_variant(variant)
    else:
        software.clean_work_dir()
