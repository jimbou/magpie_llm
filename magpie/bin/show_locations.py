import argparse
import configparser
import pathlib

import magpie

# ================================================================================
# Main function
# ================================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Magpie show locations')
    parser.add_argument('--scenario', type=pathlib.Path, required=True)
    parser.add_argument('--filename', type=str)
    parser.add_argument('--tag', type=str)
    args = parser.parse_args()

    # read scenario file
    config = configparser.ConfigParser()
    config.read_dict(magpie.core.default_scenario)
    config.read(args.scenario)

    # setup
    magpie.core.pre_setup(config)
    magpie.core.setup(config)
    software = magpie.utils.software_from_string(config['software']['software'])(config)

    # show locations
    target_files = config['software']['target_files'].split()
    for filename in software.target_files:
        if args.filename is not None and args.filename != filename:
            continue
        msg = f'==== {filename} ===='
        if magpie.settings.color_output:
            msg = f'\033[1m{msg}\033[0m'
        print(msg)
        model = software.noop_variant.models[filename]
        for tag in model.locations:
            if args.tag is not None and args.tag != tag:
                continue
            msg = f'~~~~ {tag} ~~~~'
            if magpie.settings.color_output:
                msg = f'\033[1m{msg}\033[0m'
            print(msg)
            for loc in model.locations_names[tag]:
                print(model.show_location(tag, loc))
            print()
    software.clean_work_dir()
