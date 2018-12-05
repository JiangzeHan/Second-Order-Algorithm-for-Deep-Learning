from __future__ import division

import keras.backend as K
from keras.legacy import interfaces
from keras.optimizers import Optimizer
import tensorflow as tf
from tensorflow.python.ops.gradients_impl import _hessian_vector_product


class Cubic(Optimizer):
    def __init__(self, rho=0.1, c=0.01, **kwargs):
        super(Cubic, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(0.01, name='lr')
            self.l = K.variable(100, name='l')
            self.rho = K.variable(rho, name='rho')
            self.c = K.variable(c, name='c')
            self.epsilon = K.variable(100) / self.rho

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        for p, g in zip(params, grads):
            g_norm = tf.norm(g)

            l_ = self.l * self.l / self.rho

            def true_fn():
                b_g = _hessian_vector_product(loss, [p], [g])[0]
                r = tf.reduce_sum(tf.multiply(g, b_g)) / (self.rho * g_norm * g_norm)
                r_c = -r + tf.sqrt(r * r + 2 * g_norm / self.rho)
                return -r_c * g / g_norm

            def false_fn():
                delta = tf.zeros(g.shape)
                sigma = self.c * tf.sqrt(self.epsilon * self.rho) *self.lr
                eta = self.lr / 20
                zeta = tf.random_uniform(shape=g.shape, dtype=tf.float32)
                g_ = g + sigma * zeta

                b_delta = _hessian_vector_product(loss, [p], [delta])[0]
                delta = delta - eta * (g_ + eta * b_delta + self.rho * tf.norm(delta) * delta / 2)

                b_delta = _hessian_vector_product(loss, [p], [delta])[0]
                delta = delta - eta * (g_ + eta * b_delta + self.rho * tf.norm(delta) * delta / 2)

                b_delta = _hessian_vector_product(loss, [p], [delta])[0]
                delta = delta - eta * (g_ + eta * b_delta + self.rho * tf.norm(delta) * delta / 2)

                b_delta = _hessian_vector_product(loss, [p], [delta])[0]
                delta = delta - eta * (g_ + eta * b_delta + self.rho * tf.norm(delta) * delta / 2)

                b_delta = _hessian_vector_product(loss, [p], [delta])[0]
                delta = delta - eta * (g_ + eta * b_delta + self.rho * tf.norm(delta) * delta / 2)

                b_delta = _hessian_vector_product(loss, [p], [delta])[0]
                delta = delta - eta * (g_ + eta * b_delta + self.rho * tf.norm(delta) * delta / 2)

                b_delta = _hessian_vector_product(loss, [p], [delta])[0]
                delta = delta - eta * (g_ + eta * b_delta + self.rho * tf.norm(delta) * delta / 2)

                b_delta = _hessian_vector_product(loss, [p], [delta])[0]
                delta = delta - eta * (g_ + eta * b_delta + self.rho * tf.norm(delta) * delta / 2)

                b_delta = _hessian_vector_product(loss, [p], [delta])[0]
                delta = delta - eta * (g_ + eta * b_delta + self.rho * tf.norm(delta) * delta / 2)

                b_delta = _hessian_vector_product(loss, [p], [delta])[0]
                delta = delta - eta * (g_ + eta * b_delta + self.rho * tf.norm(delta) * delta / 2)

                return delta

            v = tf.cond(tf.greater_equal(g_norm, l_), true_fn, false_fn)

            new_p = p + v

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'rho': float(K.get_value(self.rho)),
                  'c': float(K.get_value(self.c))}
        base_config = super(Cubic, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
