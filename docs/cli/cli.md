# FRECKLL Command Line Interface


The FRECKLL command line interface (CLI) allows you to solve for atmospheric chemistry using standard `yaml` input files.

Example files can be found [here](https://github.com/ahmed-f-alrefaie/freckll/tree/master/examples/inputs)


## Usage

``freckll`` is a command line tool that can be run from the terminal. The basic usage is:


```bash
freckll [OPTIONS] INPUT
```

The help message can be printed using the `-h` or `--help` flag:

```bash

Usage: freckll [OPTIONS] INPUT

  Run the Freckll simulation.

  Args:     
  input: Path to the input file.     
  output: Path to the output file.     
  overwrite: Overwrite the output file if it exists.     
  plot: Plot the results.     
  plot_path: Path to save the plots.     
  plot_prefix: Prefix for the plot filenames.     
  animate: Animate the results.

Options:
  -o, --output PATH   Output file to write.  [required]
  --overwrite         Overwrite the output file if it exists.
  --plot              Plot the results.
  --plot-path PATH    Path to save the plots.
  --plot-prefix TEXT  Prefix for the plot filenames.
  --animate           Animate the results.
  --help              Show this message and exit.
```

## CLI options

### `-o` or `--output`

Running one of the example files can be done using the following command:   

```bash
freckll -o result.hdf5 example.yaml
```

The output file will be saved as `result.hdf5` in the current directory. If the file already exists then FRECKLL will refuse to run unless the `--overwrite` flag is used.

You can read the hdf5 file using the `read_h5py_solution` function from the `freckll.io.output` module. Here is an example of how to do this:

```python

from freckll.io.output import read_h5py_solution

solution_h5 = read_h5py_solution("solution_equil.h5")
```


### Plotting

freckll can plot the results using the `--plot` flag. The plots will be saved in the current directory with the prefix `freckll_`. The plot will be saved as a png file. The `--plot-path` option can be used to specify a different directory to save the plots. The `--plot-prefix` option can be used to specify a different prefix for the plot filenames.

```bash
freckll -o result.hdf5 example.yaml --plot --plot-path ./plots --plot-prefix my_plot
``` 

### Animation

freckll can animate the results using the `--animate` flag. The animation will be saved in the current directory with the prefix `freckll_`. The animation will be saved as a mp4 file. The `--plot-path` option can be used to specify a different directory to save the plots. The `--plot-prefix` option can be used to specify a different prefix for the plot filenames.

```bash
freckll -o result.hdf5 example.yaml --animate --plot-path ./plots --plot-prefix my_plot
```
The animation will be saved as `my_plot_animation.mp4` in the `./plots` directory.


## Failure

If the simulation fails (Due to solver issues, timeout, maxiter limits etc), the output file will be saved with the suffix `_failed.hdf5`. Additionally, if plots and animations are requested, they will be saved with the suffix `_failed.png` and `_failed.mp4` respectively. The output file will contain the last state of the simulation before it failed. This can be useful for debugging and understanding what went wrong.

