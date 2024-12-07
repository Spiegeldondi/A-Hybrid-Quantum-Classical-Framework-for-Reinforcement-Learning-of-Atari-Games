from tf_agents.agents.dqn import dqn_agent
import tensorflow as tf
from tf_agents.utils import eager_utils

class CustomDqnAgent(dqn_agent.DqnAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _train(self, experience, weights):
        with tf.GradientTape() as tape:
            loss_info = self._loss(
                experience,
                td_errors_loss_fn=self._td_errors_loss_fn,
                gamma=self._gamma,
                reward_scale_factor=self._reward_scale_factor,
                weights=weights,
                training=True,
            )

        tf.debugging.check_numerics(loss_info.loss, 'Loss is inf or nan')

        variables_to_train = self._q_network.trainable_weights
        grads = tape.gradient(loss_info.loss, variables_to_train)
        grads_and_vars = list(zip(grads, variables_to_train))

        if self._gradient_clipping is not None:
            grads_and_vars = eager_utils.clip_gradient_norms(
                grads_and_vars, self._gradient_clipping)

        # Apply Layerwise Optimization
        for opt, g_v in zip(self._optimizer, grads_and_vars):
            opt.apply_gradients([g_v])

        self.train_step_counter.assign_add(1)
        self._update_target()

        return loss_info

