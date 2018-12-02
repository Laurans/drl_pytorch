from core.monitors import Monitor
from core.agents import AGENT_DICT
from core.utils import MonitorParams
from core.models import MODEL_DICT
from core.memories import MEMORY_DICT
from core.envs import ENV_DICT

import click

@click.group()
def cli():
    pass

@cli.command()
@click.option('--verbose', 'verbose', type=int, default=0, help='0 for nothing in stream | 1 for printing info in stream + file | 2 for debug')
@click.option('--machine', 'machine', type=str, default='machine', help='Machine name, used for creating a signature log file')
@click.option('--ts', 'timestamp', type=str, default='0000', help='Timestamp/number/id used for creating a signature log file')
@click.option('--vis', 'visualize', is_flag=True, help='Visualize metrics/plots with visdom')
@click.option('--render', 'env_render', is_flag=True, help='Save environment render in imgs/ dir')
@click.option('--config', 'config_number', type=int, default=0, help='Choose config from config.yaml to run')
def train(**args):
    click.echo(f'{args}')
    options = MonitorParams(**args) 

    monitor = Monitor(
        monitor_param=options,
        agent_prototype=AGENT_DICT[options.agent_type],
        model_prototype=MODEL_DICT[options.model_type],
        memory_prototype=MEMORY_DICT[options.memory_type],
        env_prototype=ENV_DICT[options.env_type],
    )

    monitor.train()

@cli.command()
def test():
    click.echo('Testing ...')

if __name__ == '__main__':
    cli()