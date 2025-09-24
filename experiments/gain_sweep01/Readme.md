# Gain Sweep Experiment Data

This path contains the data for Figure 1, representing a comprehensive gain sweep experiment across multiple algorithms.

## Experiment Parameters

### Gain Values (20 steps)
- Minimum gain: 0.025
- Maximum gain: 20.0
- Intermediate values: 
  - 0.03554137
  - 0.05052757
  - 0.07183277
  - 0.10212142
  - 0.14518143
  - 0.2063979
  - 0.2934266
  - 0.41715138
  - 0.59304534
  - 0.84310586
  - 1.19860564
  - 1.70400367
  - 2.4225053
  - 3.44396672
  - 4.89613243
  - 6.96061104
  - 9.89558733
  - 14.06811098

## Algorithms Tested
Four different implementations were evaluated:
1. gaussian1
2. gaussian2
3. ML-PWS
4. PWS (with known model)

Each algorithm was tested with identical gain values for direct comparison.

## Directory Structure
Each subfolder follows the naming convention:
`{algorithm}_gain={value}/`
containing the corresponding experimental results.
