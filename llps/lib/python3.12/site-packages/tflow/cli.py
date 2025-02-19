import sys

from .cli_args import Arg


args = [
    Arg(
        name='name',
        data_type='str',
        required=True,
    ),
    Arg(
        name='version',
        data_type='int',
        required=True,
    ),
    Arg(
        name='production',
        data_type='bool',
        required=False,
        default=False,
    ),
]


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def main():
    """
    Initialize args and renders file.
    Use sys.exit to terminate program.
    """

    # parse cli args
    from .cli_args import parse_args, ValidationError
    try:
        parsed_args = parse_args(args)
        print(parsed_args)
    except ValidationError as e:
        eprint('Argument error: {msg}'.format(msg=e))
        sys.exit(1)

    name = parsed_args['name']
    version = parsed_args['version']
    production = parsed_args['production']
    
    # render CI file
    from .render import render
    render(
        name=name,
        version=version,
    )

    import subprocess
    # git tag current commit with version, then push.
    subprocess.run(['git', 'commit', '.gitlab-ci.yml',
                    '-m', 'Bumped modelversion to {v}'.format(v=version)])
    subprocess.run(['git', 'tag', str(version)])
    subprocess.run(['git', 'push', 'origin', 'master' if production else 'develop'])
    subprocess.run(['git', 'push', '--tags'])
    
    sys.exit(0)
