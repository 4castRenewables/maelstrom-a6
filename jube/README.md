# Using JUBE for benchmarking

[JUBE](https://apps.fz-juelich.de/jsc/jube/jube2/docu/) can be used for benchmarking.
The benchmarks are defined in `submit.yaml`.

## Loading JUBE on JSC

```bash
module load JUBE/2.5.1
```

## Running the Benchmarks

To run the benchmarks, use

```bash
jube run jube/submit.yaml --tag jwc test
```

Replace the tags with the respective tags.
Available tags:

* jwc (JUWELS Cluster)
* jwb (JUWELS Booster)
* dc-gpu (JURECA NVIDIA A100)
* dc-h100 (JURECA NVIDIA H100)
* dc-mi200 (JURECA AMD MI250x)
* e4-a2 (E4 NVIDIA A2)
* e4-gh (E4 NVIDIA Grace Hopper)

### Other tags

* test (single node, devel queues, 1 epoch)
* cscratch (use `$CSCRATCH` for reading data)
* single-gpu: use single node only and submits to dc-gpu and dc-h100


*Note:*
For debugging consider the `--debug`, `--devel`, and/or `-v` options.

Once all runs are finnished, analysis can be performed via

```bash
jube result jube/ap6-run/ --id <benchmark IDs> --analyse --update jube/submit.yaml > benchmark-results.md
```
