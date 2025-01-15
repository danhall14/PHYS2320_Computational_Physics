# Computing 2 Coursework - py21spqr.py

## Overview

This project is part of the **Computing 2 Coursework** module at the University of Leeds. The script processes and analyses experimental data to calculate asymmetry between two energy channels. It fits the asymmetry data to a model to estimate key parameters, including magnetic field strength, detector angle, and damping time constant.

## Features

- **Data Loading**: Efficiently loads and preprocesses experimental data from a specified file format.
- **Data Processing**: Organises data into distinct energy channels and computes asymmetry values for analysis.
- **Asymmetry Calculation**: Computes asymmetry using histogram methods for a specified number of bins.
- **Model Fitting**: Utilises curve fitting techniques to estimate parameters such as magnetic field strength (B), detector angle (Beta), and damping time constant (Tau).
- **Visualisation**: Generates comprehensive plots, including histograms and asymmetry curves with fitted parameters, to facilitate data interpretation.

## File Structure

- `data_analysis.py`: The main Python script for processing and analysing the data.
- `data_file.dat`: The input data file containing time and channel measurements.

## Requirements

- Python 3.x
- `numpy` for numerical operations
- `matplotlib` for data visualisation
- `scipy` for optimization and curve fitting

You can install the required dependencies using:

```bash
pip install numpy matplotlib scipy
