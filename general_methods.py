"""
Luk Fleddermann
30.11.21
-----------
This file contains general (usefull) classes and functions, 
which can be used in a more general application than the 
simulation and prediction of spacially extended systems. 
"""
class progress_featback(object):
    """
    The class includes different featback functions which can be used to obtain featback for itterative aplications of similar steps.
    Some of which are reused from userbased function definitions.
    """
    def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r",name=None):
        """
        Args:
            - iteration   - Required: current iteration (Int)
            - total       - Required: total iterations (Int)
            - prefix      - Optional: prefix string (Str)
            - suffix      - Optional: suffix string (Str)
            - decimals    - Optional: positive number of decimals in percent complete (Int)
            - length      - Optional: character length of bar (Int)
            - fill        - Optional: bar fill character (Str)
            - printEnd    - Optional: end character (e.g. "\r", "\r\n") (Str)

        return:
            - none
        notes:
            - The function returns in the last step similtaniously 100% and 100%-10^(-decimals)*1%.
            - The function is a modified version of the function from the surce:
            https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console (last visited 30.11.21).
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        if name is not None:
            print(f'\r{name}\t{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
        else:
            print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
        # Print New Line on Complete
        if iteration == total-1: 
            percent = ("{0:." + str(decimals) + "f}").format(100)
            filledLength = length
            bar = fill * filledLength + '-' * (length - filledLength)
            if name is not None:
                print(f'\r{name}\t{prefix} |{bar}| {percent}% {suffix}')
            else:
                print(f'\r{prefix} |{bar}| {percent}% {suffix}')
        
    def print_percent_of_saved_frames(i, n):
        """
        Args:
            - iteration   - Required  : current iteration (Int)
            - total       - Required  : total iterations (Int)
        return:
            - none
        """
        if (100*i % n) == 0:
            print(f'Saved {100*i/n} % of the frames.', end = printEnd)