# -*- coding: utf-8 -*-
"""
INSTRUCTIONS
~~~~~~~~~~~~

This is a template file for you to use for your Computing 2 Coursework.

Save this file as py21spqr.py when py21spqr should be replaced with your IT username

Do not rename or remove the function ProcessData, the marking will assume that this function exists
and will take in the name of a data file to open and process. It will assume that the function will return
all the answers your code finds. The results are returned as a dictionary.

Your code should also produce the same plots as you produce in your report.

Your module's docstring replaces this docstring.
"""
# Do your imports here
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import warnings


def load_data(filename):
    # Define any functions you use in your code here
    """
    Load the data from the file.

    Args:
        filename (str):
            Name of the file to be read.

    Returns:
        data (ndarray):
            Numpy array with data from the file.
    """
    try:
        with open(filename, 'r') as file:
            # Ignore the header until the end marker is reached
            for line in file:
                if '&END' in line:
                    break
            else:
                raise ValueError("Could not find '&END' in file")
            # Read the data into a numpy array
            data = np.genfromtxt(file, delimiter='\t', skip_header=1)
        return data
    except IOError:
        raise IOError(f"Error: Could not open file {filename}")
        return None


def data_dictionary(data):
    """
    Process the data and separate it into different energy channels.

    Args:
        data (array):
            Array of data to be processed.

    Returns:
        energy_dict (dict):
            A dictionary with energy as key and time-channel pairs
            as values.
    """
    try:
        list_of_energies = [5, 10, 15, 20, 25]
        times = data[:, ::2]
        channels = data[:, 1::2]
        energy_dict = {}

        # Check if the maximum value of time is greater than 10 microseconds
        max_time = np.max(times)
        if max_time > 10.0:
            raise ValueError(
                f"max time value {max_time:.3f} exceeds the limit of 10.0 μs")

        # Calculate the number of columns
        num_cols = len(data[1])
        exp_num_cols = 2 * len(list_of_energies)

        # Check if the number of columns is correct
        if num_cols != exp_num_cols:
            raise ValueError(
                f"Expected {exp_num_cols} columns found {num_cols} columns")

        # Iterate over the different energy channels
        for i, energy in enumerate(list_of_energies):
            time_column = times[:, i]
            channel_column = channels[:, i]
            energy_dict[f"{energy}keV"] = np.transpose(
                np.array([time_column, channel_column]))
        return energy_dict
    except TypeError:
        raise TypeError("Input data must be a 2D numpy array")


def calc_asymmetry(data, nbins):
    """
    Calculate the asymmetry of the data.

    Args:
        data (array):
            Array of the data to be processed
        nbins (int):
            Number of bins for the histogram.

    Returns:
        A0 (array):
            Calculated asymmetry
        bin_centres (array):
            Calculated bin centres times
        err_A0 (array):
            Calculated error in asymmetry
        left_times (array):
            Array of left times (Channel = 1)
        right_times (array):
            Array of right times (Channel = 2).
    """
    times = data[:, 0]
    channels = data[:, 1]
    left_times = times[channels == 1]
    right_times = times[channels == 2]

    # Compute the histograms and errors
    Pl, _ = np.histogram(left_times, bins=nbins)
    Pr, _ = np.histogram(right_times, bins=nbins)
    error_Pl = np.sqrt(Pl)
    error_Pr = np.sqrt(Pr)

    # Compute the asymmetry and error
    err_A0 = 2 * np.sqrt((1/((Pl + Pr)**4)) *
                         ((Pl**2)*(error_Pr**2) + ((Pr**2)*(error_Pl**2))))

    A0 = (Pl - Pr) / (Pl + Pr)

    # Compute the bin edges, widths and centres
    bin_edges = np.linspace(np.min(left_times), np.max(left_times), nbins+1)
    bin_widths = np.diff(bin_edges)
    bin_centres = bin_edges[:-1] + bin_widths/2
    return A0, bin_centres, err_A0, left_times, right_times


def asymmetry_plots(A0, bin_centres, err_A0, left_times, right_times, nbins):
    """
    Plot the left and right channel histograms and the asymmetry plot.

    Args:
        A0 (array):
            Asymmetry values
        bin_centres (array):
            Bin centre times in μs
        err_A0 (array):
            Error in the asymmetry values
        left_times (array):
            Left times in μs
        right_times (array):
            Right times in μs
        nbins (int):
            Number of bins for the histograms.

    Returns:
        None
    """
    # plot the left channel histogram
    plt.figure()
    plt.hist(left_times, nbins, facecolor='k')
    plt.title('Left channel histogram for py21dh')
    plt.xlabel('Time (\u03BCs)')
    plt.ylabel('Counts')
    plt.xlim(0, 10)

    # plot the right channel histogram
    plt.figure()
    plt.hist(right_times, nbins, facecolor='k')
    plt.title('Right channel histogram for py21dh')
    plt.xlabel('Time (\u03BCs)')
    plt.ylabel('Counts')
    plt.xlim(0, 10)

    # plot the asymmetry plot
    plt.figure()
    plt.errorbar(bin_centres, A0, yerr=err_A0, color='k',
                 fmt='.', label='Measured Asymmetry')
    plt.title('Measured and fitted asymmetry for 10.0keV for py21dh')
    plt.xlabel('Time (\u03BCs)')
    plt.ylabel('Asymmetry')
    plt.xlim(0, 10)
    return


def calc_average_amp(bin_centres, A0, err_A0):
    """
    Calculate the average amplitude from the asymmetry data.

    Args:
        bin_centres (array):
            Bin centre times
        A0 (array):
            Asymmetry values
        err_A0 (array):
            Error in the asymmetry values.

    Returns:
        A0_zero_times (array):
            The times at which the asymmetry is zero in μs
        average_amp (float):
            Calculated average amplitude.
    """
    # find the times at which the asymmetry is zero
    A0_zero_indices = (np.argwhere(np.diff(np.sign(A0))).flatten())[0:2]
    A0_zero_times = bin_centres[A0_zero_indices]

    # calculate the average amplitude
    amplitude_at_start = A0[0]
    peak_error = err_A0[0]
    average_amp = amplitude_at_start - peak_error
    return A0_zero_times, average_amp


def solve_Beta_guess(Starting_Beta, average_amp):
    """
    Solve for Beta_guess using Starting_Beta and average_amp.

    Args:
        Starting_Beta (float):
            Starting estimate for Beta_guess (0.8)
        average_amp (float):
            The average amplitude.

    Returns:
        Beta_guess (float):
            Calculated Beta_guess value.
    """
    # function to solve for Beta_guess using fsolve
    return np.sin(Starting_Beta)/Starting_Beta - (3 * average_amp)


def solve_Beta_guess_equation(Starting_Beta, average_amp):
    """
    Solve the Beta_guess equation using fsolve.

    Args:
        Starting_Beta (float):
            Starting value for Beta_guess (0.8)
        average_amp (float):
            The average amplitude.

    Returns:
        Beta_guess (float):
            The solved Beta_guess value.
    """
    # use fsolve to find Beta_guess that solves the equation
    Beta_guess = fsolve(solve_Beta_guess, Starting_Beta,
                        args=(average_amp,))[0]
    return Beta_guess


def initial_B_guess(A0_zero_times):
    """
    Calculate initial guess for the magnetic field strength.

    Args:
        A0_zero_times (array):
            Array of A0 values at zero time in μs.

    Returns:
        B_guess (float):
            Initial guess for the magnetic field strength, B.
    """
    # calculate the time period and Larmor frequency
    time_period = np.mean((np.diff(A0_zero_times)) * 2)
    gamma = 851.616
    # calculate the initial guess for the magnetic field
    lamor_frequency = (2 * np.pi)/time_period
    all_B_guess = lamor_frequency/gamma
    B_guess = np.mean(all_B_guess)
    return B_guess


def fit_function(bin_centres, B, Beta, Tau):
    """
    Calculate the asymmetry of the signal as a function of time.

    Args:
        bin_centres (array):
            The bin centre times in μs
        B (float):
            The magnetic field strength, B
        Beta (float):
            The detector angle
        Tau (float):
            The damping time constant.

    Returns:
        (array):
            Calculated asymmetry values.
    """
    gamma = 851.616
    top = np.sin((gamma * B * bin_centres) - Beta) - \
        np.sin((gamma * B * bin_centres) + Beta)
    Damping = (np.exp((-bin_centres)/Tau))
    return ((-1/3) * (top / (2 * Beta))) * Damping


def fit_data(A0, bin_centres, err_A0, B_guess, Beta_guess, Tau_guess):
    """
    Fits the data to the function and plots the fit.

    Args:
        A0 (array):
            The asymmetry values
        bin_centres (array):
            The bin centre times in μs
        err_A0 (array):
            Error values for the asymmetry
        B_guess (float):
            Initial guess for the magnetic field strength, B
        Beta_guess (float):
            The initial guess for the detector angle in rad
        Tau_guess (float):
            The initial guess for damping constant in microseconds.

    Returns:
        Tuple: A tuple containing the following elements:
            popt (array):
                Optimal values for the fit parameters
            pcov (array):
                Covariance matrix for the fit parameters
            B (float):
                Final value of the magnetic field parameter
            B_unc (float):
                Uncertainty for the final magnetic field parameter value
            Beta (float):
                Final value of the detector angle parameter
            Beta_unc (float):
                Uncertainty for the final detector angle parameter value
            Tau (float):
                Final value of the damping time constant parameter
            Tau_unc (float):
                Uncertainty for the final damping time constant value.

    Raises:
        ValueError:
            If the fitting process fails.
    """
    try:
        # fit the function to the data and plot the fit
        popt, pcov = curve_fit(fit_function, bin_centres, A0, p0=[
            B_guess, Beta_guess, Tau_guess], sigma=err_A0, absolute_sigma=True)
        B = popt[0]
        B_unc = np.sqrt(pcov[0][0])
        Beta = np.abs(popt[1])
        Beta_unc = np.abs(np.sqrt(pcov[1][1]))
        Tau = popt[2]
        Tau_unc = np.sqrt(pcov[2][2])
        if B < 0 or B > 0.03:
            warnings.warn('Magnetic field B should be positive less than 30mT')
        if Beta < 0.5 or Beta > 1.5:
            warnings.warn(
                'Detector angle should be between 0.5 rad and 1.5 rad')
        if Tau < 2 or Tau > 10:
            warnings.warn(
                'Damping time constant should be between 2 us and 10 us')
        return popt, pcov, B, B_unc, Beta, Beta_unc, Tau, Tau_unc
    except ValueError as err:
        raise ValueError(f"Fit failed: {err}")
        return None


def Plots_for_10keV(bin_centres, popt, B, B_unc, Beta, Beta_unc, Tau, Tau_unc):
    """
    Plot the asymmetry data and the fit for 10 keV energy.

    Args:
        bin_centres (array):
            1D array containing the bin centre times of the asymmetry values.
        popt (array):
            Optimal values for the fit parameters
        pcov (array):
            Covariance matrix for the fit parameters
        B (float):
            Final value of the magnetic field parameter
        B_unc (float):
            Uncertainty for the final magnetic field parameter value
        Beta (float):
            Final value of the detector angle parameter
        Beta_unc (float):
            Uncertainty for the final detector angle parameter value
        Tau (float):
            Final value of the damping time constant parameter
        Tau_unc (float):
            Uncertainty for the final damping time constant parameter value.

    Returns:
        None
    """
    plt.plot(bin_centres, fit_function(bin_centres, *popt), 'r-', label='Fit')
    plt.legend()
    plt.text(0, 0.1, f'B = {(B*10**3):.3f} ± {(B_unc*10**3):.3f}mT, $\\beta$ = {(Beta):.2f} ± {(Beta_unc):.2f}rad, $\\tau$ = {(Tau):.1f} ± {(Tau_unc):.1f}$\\mu$s',
             transform=plt.gca().transAxes, fontsize=11)

    return


def quad_fit_function(Implantation_energies, a, b, c):
    """
    Calculate the quadratic fit for the magnetic field.

    Args:
        Implantation_energies (array):
            The implantation energies in keV
        a (float):
            The quadratic energy coefficient
        b (float):
            The linear energy coefficient
        c (float):
            The constant energy term.

    Returns:
        (array):
            Calculated magnetic field values.
    """
    return (a * (Implantation_energies) ** 2) + (b * (Implantation_energies)) + c


def quad_fit(Implantation_energies, B_values, B_values_unc):
    """
    Fits the magnetic field values to a quadratic function of energy.

    Args:
        Implantation_energies (array):
            An array of implantation energies in keV.
        B_values (array):
            An array of magnetic field values, B, in T.
        B_values_unc (array):
            An array of uncertainties in magnetic field values, B, in T.

    Returns:
        (tuple):
            The quadratic fit coefficients and their uncertainties.
    """
    # plot values of B against implantation energies
    plt.figure()
    plt.errorbar(Implantation_energies, B_values, yerr=B_values_unc,
                 color='k', fmt='.', label='Error in B')

    # Fit magnetic field to quadratic function
    popt, pcov = curve_fit(quad_fit_function, Implantation_energies,
                           B_values, sigma=B_values_unc, absolute_sigma=True)

    # Generate quadratic fit points
    a, b, c = popt
    a_unc, b_unc, c_unc = np.sqrt(pcov[0][0]), np.sqrt(
        pcov[1][1]), np.sqrt(pcov[2][2])
    x_data_fit = np.linspace(min(Implantation_energies),
                             max(Implantation_energies), 100)
    y_data_fit = quad_fit_function(x_data_fit, a, b, c)

    plt.plot(x_data_fit, y_data_fit, '-', label='Quadratic Fit')
    plt.xlabel('Implantation Energy (keV)')
    plt.ylabel('Magnetic Field (T)')
    plt.title('Field Profile with Energy for py21dh')
    plt.legend()
    plt.text(0.50, 0.20, f'a = {(a*10**6):.2f} ± {(a_unc*10**6):.2f} $\\mu$TkeV$^{{-2}}$\n b = {(b*10**6):.0f} ± {(b_unc*10**6):.0f}$\\mu$TkeV$^{{-1}}$\n c = {(c*10**6):.0f} ± {(c_unc*10**6):.0f}$\\mu$T',
             transform=plt.gca().transAxes, fontsize=12)
    return a, b, c, a_unc, b_unc, c_unc


def ProcessData(filename):
    """
    Processes the input data file and returns a dictionary of results.

    Parameters:
        filename (str):
            The name of the input data file.

    Returns:
        dict: A dictionary of results containing the following keys:
            "10keV_B" (float):
                The magnetic field for 10 keV data in T
            "10keV_B_error" (float):
                The error in the magnetic field for 10 keV data in T
            "beta" (float):
                The detector angle in radians
            "beta_error" (float):
                The uncertainty in detector angle in radians
            "10keV_tau_damp" (float):
                The damping time for 10 keV data in s
            "10keV_tau_damp_error" (float):
                The uncertainty in damping time for 10 keV data in s
            "B(Energy)_coeffs" (tuple):
                A tuple of coefficients (a, b, c) for quadratic, linear and constant terms
                for fitting magnetic field dependence on energy (T/keV^2, T/keV, T)
            "B(Energy)_coeffs_errors" (tuple):
                A tuple of uncertainties in above coefficients in the same order (T/keV^2, T/keV, T).
    """
    # Load data from file
    data = load_data(filename)

    # Process data and calculate energy_dict
    energy_dict = data_dictionary(data)

    # Calculate asymmetry and average amplitude values
    nbins = 500
    A0, bin_centres, err_A0, left_times, right_times = calc_asymmetry(
        data, nbins)
    A0_zero_times, average_amp = calc_average_amp(bin_centres, A0, err_A0)

    # Calculate initial guesses for B, Beta, and Tau
    Starting_Beta = 0.8

    Tau_guess = 6

    # initialise empty array to store B values and B errors
    B_values = np.array([])
    B_values_unc = np.array([])
    for energy, data in energy_dict.items():
        A0, bin_centres, err_A0, left_times, right_times = calc_asymmetry(
            data, nbins)
        A0_zero_times, average_amplitude = calc_average_amp(
            bin_centres, A0, err_A0)

        # calculate beta_guess and B_guess for current energy
        B_guess = initial_B_guess(A0_zero_times)
        Beta_guess = solve_Beta_guess_equation(
            Starting_Beta, average_amplitude)

        # fit the data for current energy
        popt, pcov, B, B_unc, Beta, Beta_unc, Tau, Tau_unc = fit_data(
            A0, bin_centres, err_A0, B_guess, Beta_guess, Tau_guess)
        B_values = np.append(B_values, B)
        B_values_unc = np.append(B_values_unc, B_unc)
        # Generate plots for 10keV energy
        if energy == '10keV':
            asymmetry_plots(A0, bin_centres, err_A0,
                            left_times, right_times, nbins)
            Plots_for_10keV(bin_centres, popt, B, B_unc,
                            Beta, Beta_unc, Tau, Tau_unc)
            B_10 = popt[0]
            B_10_unc = np.sqrt(pcov[0][0])
            Beta_10 = np.abs(popt[1])
            Beta_10_unc = np.sqrt(pcov[1][1])
            Tau_10 = popt[2]*(10**(-6))
            Tau_10_error = (np.sqrt(pcov[2][2]))*(10**(-6))

    # Fit B values as a function of energy using quadratic polynomial
    Implantation_energies = np.array([5, 10, 15, 20, 25])
    a, b, c, a_unc, b_unc, c_unc = quad_fit(
        Implantation_energies, B_values, B_values_unc)

    # This is the dictionary of results. Replace the None values with variables containing your answers
    # Do NOT remove or rename any of the dictionary entries, if you code does not find an answer, leave it
    # returning None here. Take Note of the units in the comments!

    results = {"10keV_B": B_10,  # this would be the magnetic field for 10keV data (T)
               # the error in the magnetic field (T)
               "10keV_B_error": B_10_unc,
               "beta": Beta_10,  # Detector angle in radians
               # uncertainity in detector angle (rad)
               "beta_error": Beta_10_unc,
               "10keV_tau_damp": Tau_10,  # Damping time for 10keV (s)
               "10keV_tau_damp_error": Tau_10_error,  # and error (s)
               # tuple of a,b,c for quadratic,linear and constant terms
               "B(Energy)_coeffs": (a, b, c),
               # for fitting B dependence on energy
               # (T/keV^2,T/keV,T)
               # Errors in above in same order.
               "B(Energy)_coeffs_errors": (a_unc, b_unc, c_unc),
               }
    return results


if __name__ == "__main__":
    # Put your test code in side this if statement to stop it being run when you import your code
    # Please avoid using raw_input as the testing is going to be done by a computer programme, so
    # can't input things from a keyboard....
    filename = "/Users/danhall/Desktop/Computing 2/assessment_data_py21dh.dat"
    test_results = ProcessData(filename)
    print(test_results)
