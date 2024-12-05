# Computing 2 Coursework - py21spqr.py

## Overview

This script is the implementation for the **Computing 2 Coursework** module at the University of Leeds. The objective is to process and analyse experimental data, calculate the asymmetry between two energy channels, fit the asymmetry data to a model, and estimate relevant parameters, including magnetic field strength, detector angle, and damping time constant.

## Features

- **Data Loading**: Loads experimental data from a file and processes it into a usable format.
- **Data Processing**: Separates data into different energy channels and computes asymmetry values.
- **Asymmetry Calculation**: Calculates the asymmetry of the data for a given number of bins.
- **Fitting**: Fits the asymmetry data to a model to estimate the magnetic field strength (B), detector angle (Beta), and damping time constant (Tau).
- **Plotting**: Plots histograms of the left and right channel data, as well as the asymmetry curve with the fitted parameters.

## File Structure

- `data_analysis.py`: The main Python script for processing and analysing the data.
- `data_file.dat`: The input data file containing time and channel measurements.

## Requirements

- Python 3.x
- `numpy` for numerical operations
- `matplotlib` for plotting
- `scipy` for optimization and fitting

You can install the required dependencies using:

```bash
pip install numpy matplotlib scipy
