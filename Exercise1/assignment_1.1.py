import numpy as np
import numpy.typing as npt


def load_grades(filename: str) -> npt.NDArray:
    # reads grades from the file.
    # ====================================================================
    ls = []
    with open(filename, 'r') as file:
        for line in file:
            ls.append(float(line))
    
    grades = np.array(ls, dtype=float)
    # ====================================================================
    return grades


def python_compute(array: npt.NDArray) -> tuple[float, float]:
    # computes the mean and the variance using standard Python.
    # ====================================================================
    mean, var = sum(array)/len(array), sum((array - (sum(array)/len(array)))**2)/len(array) 
    # ====================================================================
    return mean, var


def numpy_compute(array: npt.NDArray, ddof: int = 0) -> tuple[float, float]:
    # TODO: compute the mean and the variance using numpy.
    # ====================================================================
    mean, var = np.mean(array), np.var(array, ddof=ddof)
    # ====================================================================
    return mean, var

# assumes G.txt is in /data folder of the currrent directory
if __name__ == "__main__":
    # load the grades from the file, compute the mean and the
    # variance using both implementations and report the results.
    # ====================================================================
    arr = load_grades("data/G.txt")
    py_m, py_v = python_compute(arr)
    np_m, np_v = numpy_compute(arr)

    print(
        f"Python-only yields: \n"
        f"\tMean: {py_m} \tVariance: {py_v} \n"
        f"Numpy gives:\n"
        f"\tMean: {np_m} \tVariance: {np_v} \n"
        f"Difference is:\n"
        f"\tΔMean: {abs(py_m-np_m)} \tΔVariance: {abs(py_v-np_v)}"
    )
    # ====================================================================
