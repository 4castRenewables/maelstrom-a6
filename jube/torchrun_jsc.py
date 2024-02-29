"""
Fixed version of `torchrun` on Jülich Supercomputing Center. Requires
Slurm usage.

To use, modify your execution like the following:

Old
```shell
torchrun [...]
# or
python -m torch.distributed.run [...]
```

New
```shell
python /path/to/fixed_torch_run.py [...]
# or if `fixed_torch_run.py` is on `PYTHONPATH`
python -m fixed_torch_run [...]
```
"""

from argparse import ArgumentParser
import os
import runpy
import sys


def _as_bool(key, value):
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value == 1:
            return True
        if value == 0:
            return False
    elif isinstance(value, str):
        if value.lower() in ['1', 'true', 't', 'yes', 'y']:
            return True
        if value.lower() in ['0', 'false', 'f', 'no', 'n']:
            return False
    raise ValueError(
        f'The rendezvous configuration option {key} does not represent a '
        f'valid boolean value.'
    )


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--rdzv_endpoint', '--rdzv-endpoint')
    parser.add_argument('--rdzv_conf', '--rdzv-conf')
    parser.add_argument('--local_addr', '--local-addr')
    args = parser.parse_known_args()[0]

    endpoint = args.rdzv_endpoint
    host = (
        endpoint.rsplit(':', 1)[0]
        if endpoint
        else None
    )

    conf = args.rdzv_conf
    is_host = None
    if conf is not None:
        confs = conf.split(',')
        for (key, value) in map(lambda kv: kv.split('=', 1), confs):
            if key == 'is_host':
                is_host = _as_bool(key, value)
                break

    local_addr = args.local_addr

    return host, conf, is_host, local_addr


def fix_get_hostname(host, local_addr):
    if host and not local_addr:
        insertion_index = min(len(sys.argv), 1)
        sys.argv.insert(insertion_index, f'--local_addr={host}')


def fix_is_host(is_host, conf):
    if is_host is None:
        slurm_is_host = int(os.getenv('SLURM_PROCID') == '0')

        if not conf:
            insertion_index = min(len(sys.argv), 1)
            sys.argv.insert(
                insertion_index,
                f'--rdzv_conf=is_host={slurm_is_host}',
            )
        else:
            # Since `torchrun` only uses standard `argparse` for
            # parsing, we do not need to worry about discerning multiple
            # `--rdzv_conf` arguments (one for `torchrun`, one for the
            # script).
            for (i, arg) in enumerate(sys.argv):
                if (
                        arg.startswith('--rdzv_conf')
                        or arg.startswith('--rdzv-conf')
                ):
                    # Handle specification as two arguments vs. as one
                    # argument.
                    if arg in ['--rdzv_conf', '--rdzv-conf']:
                        modification_index = i + 1
                        old_conf = sys.argv[modification_index]
                    else:
                        modification_index = i
                        old_conf = (
                            sys.argv[modification_index].split('=', 1)[1])

                    # Handle empty conf specification.
                    if old_conf:
                        sys.argv[modification_index] = (
                            f'{sys.argv[modification_index]},')
                    sys.argv[modification_index] = (
                        f'{sys.argv[modification_index]}'
                        f'is_host={slurm_is_host}'
                    )
                    break


def main():
    host, conf, is_host, local_addr = parse_args()
    fix_get_hostname(host, local_addr)
    fix_is_host(is_host, conf)
    runpy.run_module('torch.distributed.run', run_name='__main__')


if __name__ == '__main__':
    main()
