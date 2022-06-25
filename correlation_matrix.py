"""
SStasevic, 25-04-22
Create correlation matrix for IRD reduced data.

Calculates Pearson correlation between science target + reference target frames within
a specified area.

Inputs:
    sof file containing path to science target + reference targets

Options:
    inner_radius: inner radius of area used to calculate correlation [int]
    outer_radius: outer radius of area used to calculate correlation [int]
    channels: waveband channels of IRDIS data to process [0, 1, 2 (both)]
    use_science_target: whether to include science target in frame correlation [bool]

Outputs:
    correlation matrix

"""

import argparse, warnings
import numpy as np
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning

warnings.simplefilter('ignore', category=AstropyWarning)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('sof', type=str)
    parser.add_argument('--inner_radius', type=int, default=10)
    parser.add_argument('--outer_radius', type=int, default=100)
    parser.add_argument('--channels', type=int)