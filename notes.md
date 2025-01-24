The dimensions are not tight, on this case there is an input of shape of 9, but the output is on six features:

X_input.shape
Out[26]: (9,)

Y_out.shape
Out[28]: (60, 6, 3)
(num_time_steps x num features x num_samples)

Y_out[0, :, 2]
array([7.11217510e-02, 1.31989516e-02, 1.52398373e-03, 5.67449058e+01, 3.23292363e+01, 1.68055482e+01])
