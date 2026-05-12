import numpy as np
import numpy.typing as npt


def load_grades(filename: str) -> npt.NDArray:
    # TODO: read grades from the file.
    # ====================================================================

    grades = np.loadtxt(filename, dtype=float)
    
    # ====================================================================
    return grades


def python_compute(array: npt.NDArray) -> tuple[float, float]:
    # TODO: compute the mean and the variance using standard Python.
    # ====================================================================
    n_samples = len(array)

    mean = sum(array) / n_samples

    # The worksheet defines the unbiased sample variance estimator,
    # which divides by N - 1 instead of N.
    var = sum((value - mean) ** 2 for value in array) / (n_samples - 1)
    # ====================================================================
    return mean, var


def numpy_compute(array: npt.NDArray, ddof: int = 0) -> tuple[float, float]:
    # TODO: compute the mean and the variance using numpy.
    # ====================================================================
    mean = np.mean(array)
    var = np.var(array, ddof=ddof)
    # ====================================================================
    return mean, var


if __name__ == "__main__":
    # TODO: load the grades from the file, compute the mean and the
    # variance using both implementations and report the results.
    # ====================================================================
    grades = load_grades("data/G.txt")

    python_mean, python_var = python_compute(grades)

    numpy_mean_biased, numpy_var_biased = numpy_compute(grades)
    numpy_mean_unbiased, numpy_var_unbiased = numpy_compute(grades, ddof=1)

    print("Python implementation")
    print(f"Mean:     {python_mean:.6f}")
    print(f"Variance: {python_var:.6f}")

    print("\nNumPy implementation with ddof=0")
    print(f"Mean:     {numpy_mean_biased:.6f}")
    print(f"Variance: {numpy_var_biased:.6f}")

    print("\nNumPy implementation with ddof=1")
    print(f"Mean:     {numpy_mean_unbiased:.6f}")
    print(f"Variance: {numpy_var_unbiased:.6f}")

    # ====================================================================
