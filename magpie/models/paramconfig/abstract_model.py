import magpie.core

from .realms import Realm


class AbstractConfigModel(magpie.core.BasicModel):
    def __init__(self, filename):
        super().__init__(filename)
        self.indirect_locations = False
        self.config = {
            'timing': ['run'], # setup / compile / test / run
            'cli_prefix': '--',
            'cli_glue': '=',
            'cli_boolean': 'show', # show ; hide ; prefix
            'cli_boolean_prefix_true': '',
            'cli_boolean_prefix_false': 'no-',
            'cli_none': 'hide', # show ; hide
            'silent_prefix': '@',
            'silent_suffix': '$',
        }

    def setup(self, config, section_name):
        super().setup(config, section_name)
        for name in tuple({'paramconfig', section_name}):
            config_section = config[name]
            if (k := 'timing') in config_section:
                tmp = config_section[k].split()
                if any((val := timing) not in ['setup', 'compile', 'test', 'run'] for timing in tmp):
                    msg = f'Illegal timing value: [{name}] "{val}"'
                    raise magpie.core.ScenarioError(msg)
                self.config[k] = tmp
            for k in [
                    'cli_prefix',
                    'cli_glue',
                    'cli_boolean',
                    'cli_boolean_prefix_true',
                    'cli_boolean_prefix_false',
                    'cli_none',
                    'silent_prefix',
                    'silent_suffix',
            ]:
                if k in config_section:
                    self.config[k] = config_section[k]

    def dump(self):
        return ''.join([f'{k} := {v!r}\n' for k,v in self.contents['current'].items() if not self.would_be_ignored(k, v)])

    def show_location(self, target_type, target_loc):
        if target_type != 'param':
            raise ValueError
        tag_start = ''
        tag_end = ':'
        default_start = 'default='
        if magpie.settings.color_output:
            tag_start = f'\033[36m{tag_start}'
            tag_end = f'{tag_end}\033[0m'
            default_start = f'\033[30m{default_start}\033[0m'
        space = self.contents['space'][target_loc]
        default = self.contents['current'][target_loc]
        return f'{tag_start}{target_loc}{tag_end} {space} {default_start}{default}'

    def random_value(self, key):
        realm = self.contents['space'][key]
        return Realm.random_value_from_realm(realm)

    def update_cli(self, variant, cli, step):
        if step in self.config['timing']:
            return f'{cli} {self.resolve_cli()}'
        return cli

    def resolve_cli(self):
        all_params = self.contents['current']
        tmp = [self.resolve_cli_param(k, v) for k,v in all_params.items() if not self.would_be_ignored(k, v)]
        return ' '.join([s for s in tmp if s != ''])

    def resolve_cli_param(self, param, value):
        if param.startswith(self.config['silent_prefix']):
            return ''
        prefix = self.config['cli_prefix']
        if self.config['silent_suffix'] in param:
            cli_param, *_ = param.split(self.config['silent_suffix'])
        else:
            cli_param = param
        if str(value) == 'True':
            if self.config['cli_boolean'] == 'hide':
                return f'{prefix}{cli_param}'
            if self.config['cli_boolean'] == 'prefix':
                bool_prefix = self.config['cli_boolean_prefix_true']
                return f'{prefix}{bool_prefix}{cli_param}'
        elif str(value) == 'False':
            if self.config['cli_boolean'] == 'hide':
                return ''
            if self.config['cli_boolean'] == 'prefix':
                bool_prefix = self.config['cli_boolean_prefix_false']
                return f'{prefix}{bool_prefix}{cli_param}'
        elif str(value) == 'None':
            if self.config['cli_none'] == 'hide':
                return ''
        glue = self.config['cli_glue']
        return f'{prefix}{cli_param}{glue}{value!r}'

    def would_be_ignored(self, key, value):
        return any(self.contents['current'][_k2] not in _vals for (_k1, _k2, _vals) in self.contents['conditionals'] if _k1 == key)

    def would_be_valid(self, key, value):
        for d in self.contents['forbidden']:
            forbidden = True
            for k in d:
                if (k != key and d[k] != self.contents['current'][k]) or (k == key and d[k] != value):
                    forbidden = False
                    break
            if forbidden:
                return False
        return True

    def do_set(self, target, value):
        key = target[2]
        used = self.would_be_valid(key, value) and not self.would_be_ignored(key, value)
        if used:
            self.contents['current'][key] = value
        return used
