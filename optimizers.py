from __future__ import division
import math

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.gradients_impl import _hessian_vector_product

def get_shape(shape_tuple):
    shape = 1
    for i in shape_tuple:
        shape *= i
    return shape

def cubic_gd(gradient, loss, param, epsilon, l, rho):
	# Cubic-subsolver via Gradient Descent
    grad_norm = tf.norm(gradient)

    op1 = tf.greater_equal(grad_norm, l * l / rho)

    def f1():
        b_g = _hessian_vector_product(loss, [param], [gradient])[0]
        gradient_ = tf.reshape(gradient, [get_shape(gradient.get_shape().as_list()), 1])
        b_g = tf.reshape(b_g, [get_shape(b_g.get_shape().as_list()), 1])
        r = tf.reduce_sum(tf.matmul(tf.transpose(gradient_), b_g) / (rho * grad_norm * grad_norm))
        r_c = -r + tf.sqrt(r * r + 2 * grad_norm / rho)
        delta = -r_c * gradient / grad_norm
        return delta

    def f2():
        delta = tf.Variable(np.zeros(gradient.shape), trainable=False, name='delta')
        c = 0.00001
        sigma = c * math.sqrt(epsilon * rho) / l
        eta = 1 / (20 * l)
        zeta = tf.random_uniform(shape=gradient.shape, dtype=tf.float64)
        grad = gradient + tf.cast(sigma, tf.float64) * zeta

        def cond(delta, loss, param, eta, rho):
            return tf.less(0, 1)

        def body(delta, loss, param, eta, rho):
            delta_norm = tf.norm(delta)
            b_delta = _hessian_vector_product(loss, [param], [delta])[0]
            if not b_delta:
                b_delta = tf.constant(np.zeros(gradient.shape))
            delta = delta - tf.cast(eta, tf.float64) * (grad + tf.cast(eta, tf.float64) * b_delta + tf.cast(rho, tf.float64) * delta_norm * delta / 2)
            return delta, loss, param, eta, rho   

        delta, _, _, _, _ = tf.while_loop(cond, body, loop_vars=(delta, loss, param, eta, rho), maximum_iterations=10)
        
        return delta

    delta = tf.cond(op1, f1, f2)

    return delta

def cg(gradient, global_step_tensor, gradient_, delta_):
	# Conjugate Gradient Descent Method & Gradient Descent Method:
	# Parameters:
	# gradient: current gradient
	# gradient_: gradient of former point
	# delta_: descent direction of former point
	# Return： new descent direction
    op1 = tf.equal(global_step_tensor, 0)

    def f1():
        delta = -gradient

        return delta

    def f2():
        grad_norm = tf.norm(gradient)
        grad_norm_ = tf.norm(gradient_)
        delta = -gradient + grad_norm * grad_norm / (grad_norm_ * grad_norm_) * delta_

        return delta

    delta = tf.cond(op1, f1, f2)

    return delta
