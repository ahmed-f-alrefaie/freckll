# Yaml file format

FRECKLL CLI uses the YAML file format for input files. The YAML file is a human-readable data serialization format that is easy to read and write.

For certain data types, FRECKLL yaml files can accept units in the form of strings. For example, the following are all valid:

```yaml
planet_radius: 1.0 Rjup
```

```yaml
planet_radius: 1.0 Rearth
```

```yaml
planet_radius: 20000 km
```

Generally all of the [astropy units](https://docs.astropy.org/en/stable/units/index.html) are accepted. In the documentation, we will use the following format to denote units:

- `[length]` for any length unit: e.g. `m`, `cm`, `km`, `Rjup`, `Rearth`, etc.
- `[time]` for any time unit: e.g. `s`, `min`, `hr`, `day`, `yr`, etc.
- `[mass]` for any mass unit: e.g. `kg`, `g`, `Mearth`, `Mjup`, etc.
- `[temperature]` for any temperature unit: e.g. `K`, `C`, `F`, etc.
- `[pressure]` for any pressure unit: e.g. `Pa`, `bar`, etc.
- `[spectral]` for any spectral unit: e.g. `nm`, `um`, `cm-1`, etc.
- `[flux]` for any flux unit: e.g. `W/m2/nm`, `erg/s/cm2/nm`, etc.
- `[kinetic]` for any kinematic viscosity unit: e.g. `cm2/s`, `m2/s`, etc.

Generally square units can be defined as either `[unit]2` or `[unit]**2`. For example, `m2` and `m**2` are both valid. The same applies to cube units, e.g. `m3` and `m**3`.

## Example input file

```yaml
planet:
  mass: 0.714 Mjup
  radius: 1.38 Rjup
  distance: 0.04747 AU
  albedo: 0.0

star:
  incident_angle: 45 deg
  spectrum: hd209458

# Built in format
network: venot-methanol-2020-reduced
photochemistry: venot-methanol-2020-photo

thermochemistry:
  format: ace
  abundances: [12.0, 10.925, 9.168, 7.86, 8.633]
  elements: [H, He, C, N, O]

atmosphere:
  tp_profile:
    format: from-file
    filename: tpprofile.csv
    temperature_column: 1
    pressure_column: 0
    comments: "#"
    temperature_unit: K
    pressure_unit: mbar

  kzz:
    format: from-file
    filename: kzz.csv
    kzz_column: 1
    pressure_column: 0
    comments: "#"
    kzz_unit: cm2/s
    pressure_unit: mbar

solver:
  method: rosenbrock
  t_span: [0.0, 1e10]
  max_iter: 100
  nevals: 200
  dn_crit: 1e-3
  dndt_crit: 1e-6
  max_solve_time: 1 hour
  enable_diffusion: false
  rtol: 1e-2
  atol: 1e-25
  maxiter: 100
```

We will go through each of the sections in the input file in detail. The input file is divided into the following sections:

- `planet`: This section contains the properties of the planet, such as mass, radius, distance from the star, and albedo.
- `star`: This section contains the properties of the star, such as the incident angle and spectrum.
- `network`: This section contains the name of the chemical network to be used. The network can be either a built-in network or a custom network.
- `photochemistry`: This section contains the name of the photochemistry network to be used. The network can be either a built-in network or a custom network.
- `thermochemistry`: This section contains the thermochemistry data, such as the format of the data and the abundances of the elements.
- `atmosphere`: This section contains the properties of the atmosphere, such as the temperature profile and eddy diffusion coefficient.
- `solver`: This section contains the properties of the solver, such as the method to be used, the time span, and the maximum number of iterations.

## Planet

The `planet` section contains the properties of the planet. It is a **required** section and must be present in the input file.
The following properties are available:

- `mass` - Format: `[mass]` - _Required_ - The mass of the planet in [mass] (e.g. `1 Mjup`, `2 Mearth`, etc.).
- `radius` - Format: `[length]` - _Required_ - The radius of the planet in `[length]` (e.g. `2 Rjup`, `1 Rearth`, etc.).
- `distance` - Format: `[length]` - _Required_ for photochemistry. _Optional_ otherwise. - The distance of the planet from the star in `[length]` (e.g. `1 AU`, `2 km`, etc.).
- `albedo` - Format: `float` - _Optional_ (Used in Photochemistry) - Determines how much flux reflects back from the BOA. - Default is `0.0`.

### Example

```yaml
planet:
  mass: 0.714 Mjup
  radius: 1.38 Rjup
  distance: 0.04747 AU
  albedo: 0.0
```

## Star

The `star` section contains the properties of the star. It is a **required** section if `photochemistry` is enabled.

The following properties are available:

- `incident_angle`

      - Format: `[angle]`
      - _Optional_
      - The angle of the star with respect to the planet in [angle] (e.g. `10 deg`, `1.2 rad`, etc.).
      - Default is `45.0 deg`.

- `spectrum` - Format: `string` or `rescale` or `from-file` - _Required_ if `photochemistry` is enabled. - The spectrum of the star. The spectrum can be either a built-in spectrum/custom spectrum/rescale from empirical data. - The built-in spectrum can be one of the following: `55cnc`, `adleo`, `gj436`, `gj3470`, `hd128167`, `hd189733`, `hd209458`, `sun`, `wasp12`, `wasp39`, `wasp43`

### Example

```yaml
star:
  incident_angle: 45 deg
  spectrum: hd209458
```

### from-file spectrum

The `from-file` option can be used to specify a custom spectrum file. Freckll supports a CSV file format. The arguments are as follows:

- `filename` - Format: `Path` - _Required_ - The path to the custom spectrum file. The file should be in CSV format.
- `flux_column` - Format: `int` - _Required_ - The column number of the flux data in the CSV file. The first column is `0`.
- `spectral_column` - Format: `int` - _Required_ - The column number of the spectral data in the CSV file. The first column is `0`.
- `flux_unit` - Format: `string` - _Required_ - The unit of the flux data. The unit should be in the form of a string (e.g. `W/m2/nm`, `erg/s/cm2/nm`, etc.).
- `spectral_unit` - Format: `string` - _Required_ - The unit of the spectral data. The unit should be in the form of a string (e.g. `nm`, `um`, `cm-1`, etc.).
- `reference_distance` - Format: `[length]` - _Required_ - The reference distance for the flux data in `[length]` (e.g. `1 AU`, `1 km`, etc.).
- `skiprows` - Format: `int` - _Optional_ - The number of rows to skip at the beginning of the file. Default is `0`.
- `delimiter` - Format: `string` - _Optional_ - The delimiter used in the CSV file. Default is `None`.
- `comments` - Format: `string` - _Optional_ - The character used to denote comments in the CSV file. Default is `None`.

#### Example

```yaml
star:
  incident_angle: 45 deg
  spectrum:
    format: from-file
    filename: spectrum.csv
    flux_column: 1
    spectral_column: 0
    flux_unit: W/m2/nm
    spectral_unit: nm
    reference_distance: 1 AU
    skiprows: 0
    delimiter: ","
    comments: "#"
```

### Rescale

The `rescale` option can be used to generate a custom spectrum from one of the built-in spectra. The arguments are as follows:

- `from_star` - Format: `string` - _Required_ - The name of the built-in star spectrum to be used as a reference. The built-in spectrum can be one of the following: `55cnc`, `adleo`, `gj436`, `gj3470`, `hd128167`, `hd189733`, `hd209458`, `sun`, `wasp12`, `wasp39`, `wasp43`
- `temperature` - Format: `[temperature]` - _Required_ - The temperature of the new star in `[temperature]` (e.g. `6117 K`, `3000 K`, etc.).
- `radius` - Format: `[length]` - _Required_ - The radius of the new star in `[length]` (e.g. `1.16 Rsun`, `2 Rjup`, etc.).

#### Example

```yaml
star:
  incident_angle: 45 deg
  spectrum:
    format: rescale
    from_star: sun
    temperature: 6117 K
    radius: 1.16 Rsun
```

## Network

The `network` section defines the chemical network of the simulation. It is a **required** section and must be present in the input file.

To use the built in network only requires the name of the network. For example:

```yaml
network: venot-methanol-2020-reduced
```

The available built-in networks are:

- `veillet-2024` - Full network from [Veillet et al. (2024)](https://www.aanda.org/articles/aa/full_html/2024/02/aa46680-23/aa46680-23.html)
- `venot-methanol-2020-reduced` - Reduced network for the [Venot et al. (2020)](https://www.aanda.org/articles/aa/full_html/2020/02/aa36697-19/aa36697-19.html) methanol network.
- `venot-methanol-2020` - Full network for the [Venot et al. (2020)](https://www.aanda.org/articles/aa/full_html/2020/02/aa36697-19/aa36697-19.html) methanol network.

## Custom network

If you want to use a custom network, you can specify the path to the network file. The arguments are as follows:

- `format` - Format: `string` - _Required_ - The name of the chemical network to be used. Only `venot` format is supported.

### Venot format

The `venot` format is a custom format used by FRECKLL. The arguments are as follows:

- `network_path` - Format: `Path` - _Required_ - The path to the custom network file. The file should be in the `venot` format.

#### Example

```yaml
network:
  format: venot
  network_path: path/to/network
```

## Photochemistry

The `photochemistry` section contains the name of the photochemistry network to be used. When included, photochemistry will be enabled.

Similar to the Network section, the photochemistry section can either be a built-in network or a custom network.
The following built-in photochemistry networks are available:

- `veillet-2024-photo` - Photochemistry network for the [Veillet et al. (2024)](https://www.aanda.org/articles/aa/full_html/2024/02/aa46680-23/aa46680-23.html) network.
- `venot-methanol-2020-photo` - Photochemistry network for the [Venot et al. (2020)](https://www.aanda.org/articles/aa/full_html/2020/02/aa36697-19/aa36697-19.html) methanol network.

### Example

```yaml
photochemistry: venot-methanol-2020-photo
```

## Custom photochemistry

If you want to use a custom photochemistry network. The arguments are as follows:

- `format` - Format: `string` - _Required_ - The name of the photochemistry network to be used. Only `venot` format is supported.

### Venot format

The `venot` format is a custom format used by FRECKLL. The arguments are as follows:

- `photodissociation_file` - Format: `Path` - _Required_ - The path to the custom photodissociation file. The file should be in the `venot` format.
- `cross_section_path` - Format: `Path` - _Required_ - The path to the cross-section folder. - Must include both `se*` and `qy*` files.

#### Example

```yaml
photochemistry:
  format: venot
  photodissociation_file: path/to/photodissociation
  cross_section_path: path/to/cross_section
```

## Thermochemistry

The `thermochemistry` section contains the thermochemistry data. It is a **required** section and must be present in the input file.

Currently only `ace` is supported. The arguments are as follows:

- `format` - Format: `string` - _Required_ - The name of the thermochemistry format to be used. Only `ace` format is supported.
- `elements` - Format: `list` - _Required_ - The elements in the thermochemistry data. The elements should be in the form of a list (e.g. `[H, He, C, N, O]`).
- `abundances` - Format: `list` - _Required_ - The abundances of the elements in the thermochemistry data. The abundances should be in the form of a list (e.g. `[12.0, 10.925, 9.168, 7.86, 8.633]`). - Abundances should correspond to the elements in the same order.
- `therm_file` - Format: `Path` - _Optional_ - The path to the NASA thermochemistry file. The file should be in the `ace` format. - If not defined then FRECKLLs built-in ACE file will be used.

#### Example

```yaml
thermochemistry:
  format: ace
  abundances: [12.0, 10.925, 9.168, 7.86, 8.633]
  elements: [H, He, C, N, O]
  therm_file: path/to/NASA.therm
```

## Atmosphere

The `atmosphere` section contains the properties of the atmosphere. It is a **required** section and must be present in the input file.

The following properties are available:

- `tp_profile`

      - Defines the temperature profile of the atmosphere. The arguments are as follows:
      - `format`
        - Format: `string`
        - _Required_
        - The format of the temperature profile. Only `from-file` is supported.
      - `filename`
        - Format: `Path`
        - _Required_
        - The path to the temperature profile file. The file should be in CSV format.
      - `temperature_column`
        - Format: `int`
        - _Required_
        - The column number of the temperature data in the CSV file. The first column is `0`.
      - `pressure_column`
        - Format: `int`
        - _Required_
        - The column number of the pressure data in the CSV file. The first column is `0`.
      - `comments`
        - Format: `string`
        - _Optional_
        - The character used to denote comments in the CSV file. Default is `#`.
      - `temperature_unit`
        - Format: `string`
        - _Required_
        - The unit of the temperature data. The unit should be in the form of a string (e.g. `K`, `C`, `F`, etc.).
      - `pressure_unit`
        - Format: `string`
        - _Required_
        - The unit of the pressure data. The unit should be in the form of a string (e.g. `Pa`, `bar`, etc.).
      - `skiprows`
        - Format: `int`
        - _Optional_
        - The number of rows to skip at the beginning of the file. Default is `0`.
      - `delimiter`
        - Format: `string`
        - _Optional_
        - The delimiter used in the CSV file. Default is `None`.
      - `start`
        - Format: `top` or `bottom`
        - _Optional_
        - Whether the profile starts from the top or bottom of the atmosphere. Default is `bottom`.

- `kzz` - Defines the eddy diffusion coefficient profile of the atmosphere. The arguments are as follows: - `format` - Format: `string` - _Required_ - The format of the eddy diffusion coefficient profile. Only `from-file` is supported. - `filename` - Format: `Path` - _Required_ - The path to the eddy diffusion coefficient profile file. The file should be in CSV format. - `kzz_column` - Format: `int` - _Required_ - The column number of the eddy diffusion coefficient data in the CSV file. The first column is `0`. - `pressure_column` - Format: `int` - _Required_ - The column number of the pressure data in the CSV file. The first column is `0`. - `comments` - Format: `string` - _Optional_ - The character used to denote comments in the CSV file. Default is `#`. - `kzz_unit` - Format: `string` - _Required_ - The unit of the eddy diffusion coefficient data. The unit should be in the form of a string (e.g. `cm2/s`, `m2/s`, etc.). - `pressure_unit` - Format: `string` - _Required_ - The unit of the pressure data. The unit should be in the form of a string (e.g. `Pa`, `bar`, etc.). - `skiprows` - Format: `int` - _Optional_ - The number of rows to skip at the beginning of the file. Default is `0`. - `delimiter` - Format: `string` - _Optional_ - The delimiter used in the CSV file. Default is `None`. - `start` - Format: `top` or `bottom` - _Optional_ - Whether the profile starts from the top or bottom of the atmosphere. Default is `bottom`.
- `interpolate_kzz` - Format: `bool` - _Optional_ - Whether to interpolate the eddy diffusion coefficient profile. Default is `True`.

For `kzz` it is also possible to use a constant value. e.g

```yaml
atmosphere:
  tp_profile: ...

  kzz: 1e10 cm2/s
```

### Example

```yaml
atmosphere:
  tp_profile:
    format: from-file
    filename: tpprofile.csv
    temperature_column: 1
    pressure_column: 0
    comments: "#"
    temperature_unit: K
    pressure_unit: Pa

  kzz:
    format: from-file
    filename: kzz.csv
    kzz_column: 1
    pressure_column: 0
    comments: "#"
    kzz_unit: m2/s
    pressure_unit: bar

interpolate_kzz: true
```

## Solver

The `solver` section contains the properties of the solver. It is a **required** section and must be present in the input file.
Only two solvers are currently available: `rosenbrock` and `vode`.

All solvers have these properties:

              vmr: FreckllArray,
              t_span: tuple[float, float],
              enable_diffusion: bool = False,
              atol: float = 1e-25,
              rtol: float = 1e-2,
              df_criteria: float = 1e-3,
              dfdt_criteria: float = 1e-8,

- `method`

      - Format: `string`
      - _Required_
      - The method to be used. Only `rosenbrock` and `vode` are supported.

- `t_span` - Format: `list[float,float]` - _Required_ - The start and end time of the simulation - The time span should be in the form of a list (e.g. `[0.0, 1e10]`).
- `enable_diffusion` - Format: `bool` - _Optional_ - Whether to enable molecular diffusion. Default is `False`.
- `atol` - Format: `float` - _Optional_ - The absolute tolerance of the solver. Default is `1e-25`.
- `rtol` - Format: `float` - _Optional_ - The relative tolerance of the solver. Default is `1e-3`.
- `df_criteria` - Format: `float` - _Optional_ - The criteria for convergence of the solver. - Defined as $|y_{i} - y_{i-1}| < \Delta f$. - Default is `1e-3`.
- `dfdt_criteria` - Format: `float` - _Optional_ - The criteria for convergence of the solver. - Defined as $\frac{|y_{i} - y_{i-1}|}{t_i - t_{i-1}} < \frac{\Delta f}{\Delta t}$. - Default is `1e-8`.

### Rosenbrock

The `rosenbrock` solver has the following additional properties:

- `maxiter` - Format: `int` - _Optional_ - The maximum number of iterations for the solver. Default is `100`.
- `nevals` - Format: `int` - _Optional_ - The number of evaluations to store in the solution. - These are evenly spaced in $\log_{10}(t)$. - The results are interpolated to these points. - Default is `200`.
- `timestep_reject_factor`

      - Format: `float`
      - _Optional_
      - The factor by which to reduce the timestep if the solver fails a step.
      - Default is `0.5`.

- `max_solve_time` - Format: `[time]` - _Optional_ - The maximum time to run the solver. Default is `None`

#### Example

```yaml
solver:
  method: rosenbrock
  t_span: [0.0, 1e10]
  max_iter: 100
  nevals: 200
  dn_crit: 1e-3
  dndt_crit: 1e-6
  max_solve_time: 1 hour
  enable_diffusion: false
  rtol: 1e-2
  atol: 1e-25
```

### Vode

The `vode` solver has the following additional properties:

- `max_retries` - Format: `int` - _Optional_ - The maximum number of retries for the solver. Default is `10`.
- `nevals` - Format: `int` - _Optional_ - The number of evaluations to store in the solution. - These are evenly spaced in $\log_{10}(t)$. - The results are interpolated to these points. - Default is `200`.

#### Example

```yaml
solver:
  method: vode
  t_span: [0.0, 1e10]
  max_retries: 10
  nevals: 200
  dn_crit: 1e-3
  dndt_crit: 1e-6
  max_solve_time: 1 hour
  enable_diffusion: false
  rtol: 1e-2
  atol: 1e-25
```
