import argparse
import os
from pathlib import Path


class ValidationError(Exception):
    pass


class Arg:
    def __init__(self,
        name,
        data_type,
        ensure_exists=False,
        argparse_name=None,
        argparse_dest=None,
        argparse_action=None,
        required=False,
        default=None,
    ):
        self.name = name
        self.data_type = data_type
        self.ensure_exists = ensure_exists

        if argparse_name is not None:
            self.argparse_name = argparse_name
        else:
            self.argparse_name = '--{name}'.format(name=name.replace('_', '-'))

        if argparse_dest is not None:
            self.argparse_dest = argparse_dest
        else:
            self.argparse_dest = name

        self.argparse_action = argparse_action
        if argparse_action is None:
            if data_type == 'bool':
                self.argparse_action = 'store_true'

        self.required = required
        self.default = default

    def __str__(self):
        return (
            'Arg('
            'name={self.name!r}, '
            'data_type={self.data_type!r}, '
            'argparse_name={self.argparse_name!r}, '
            'argparse_dest={self.argparse_dest!r}, '
            'argparse_action={self.argparse_action!r}, '
            'required={self.required!r}, '
            'default={self.default!r}'
            ')'
        ).format(self=self)


def parse_args(args):
    results = {}

    # use argparse to parse cli args
    parser = argparse.ArgumentParser()

    for arg in args:
        parser_kwargs = dict(
            dest=arg.argparse_dest,
        )

        if arg.argparse_action is not None:
            parser_kwargs['action'] = arg.argparse_action

        parser.add_argument(arg.argparse_name, **parser_kwargs)

    parsed_args = parser.parse_args()

    # after argparse, extend with additional functionality
    for arg in args:

        # read value from either cli arg or env var
        value = getattr(parsed_args, arg.argparse_dest, None)

        # if value is not present, and default is available, use default
        if value is None and arg.default is not None:
            value = arg.default

        # if value is required, and value not present, raise error
        if arg.required and value is None:
            raise ValidationError('Argument {name} is required'.format(
                name=arg.name
            ))

        # if there is a value, parse it according to data type
        if value is not None:
            if arg.data_type == 'str':
                value = str(value)

            if arg.data_type == 'int':
                value = int(value)

            if arg.data_type == 'bool':
                value = value in ['true', 'True', 'TRUE', '1', 'yes']

            if arg.data_type == 'path':
                value = Path(value)

        # if data_type is path and ensure_exists is true, ensure path exists
        if value is not None and arg.data_type == 'path' and arg.ensure_exists:
            if any([
                arg.ensure_exists == 'file' and not value.is_file(),
                arg.ensure_exists == 'dir' and not value.is_dir(),
            ]):
                raise ValidationError(
                    'Path does not exist: {name}={value}'.format(
                        name=arg.name,
                        value=value,
                    )
                )

        results[arg.name] = value

    return results
