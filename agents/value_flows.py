from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ValueVectorField, ActorVectorField


class ValueFlowsAgent(flax.struct.PyTreeNode):
    """Value Flows agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        """Compute the flow distributional critic loss."""
        batch_size = batch['actions'].shape[0]
        rng, actor_rng, noise_rng, time_rng, q_rng, ret_rng = jax.random.split(rng, 6)

        # Sample next actions using rejection sampling
        next_actions = self.sample_actions(batch['next_observations'], actor_rng)

        # Using target networks to compute the confidence weights.
        ret_noises = jax.random.normal(ret_rng, (batch_size, 1))
        _, ret_jac_eps_prods1 = self.compute_flow_returns(
            ret_noises, batch['observations'], batch['actions'],
            flow_network_name='target_critic_flow1', return_jac_eps_prod=True)
        _, ret_jac_eps_prods2 = self.compute_flow_returns(
            ret_noises, batch['observations'], batch['actions'],
            flow_network_name='target_critic_flow2', return_jac_eps_prod=True)
        ret_stds1 = jnp.sqrt(ret_jac_eps_prods1.squeeze(-1) ** 2)
        ret_stds2 = jnp.sqrt(ret_jac_eps_prods2.squeeze(-1) ** 2)
        if self.config['q_agg'] == 'min':
            ret_stds = jnp.minimum(ret_stds1, ret_stds2)
        else:
            ret_stds = (ret_stds1 + ret_stds2) / 2
        weights = jax.nn.sigmoid(-self.config['confidence_weight_temp'] / ret_stds) + 0.5
        weights = jax.lax.stop_gradient(weights)

        # BCFM  regularization loss
        noises = jax.random.normal(noise_rng, (batch_size, 1))
        times = jax.random.uniform(time_rng, (batch_size, 1))
        next_returns1 = self.compute_flow_returns(
            noises, batch['next_observations'], next_actions,
            flow_network_name='target_critic_flow1')
        next_returns2 = self.compute_flow_returns(
            noises, batch['next_observations'], next_actions,
            flow_network_name='target_critic_flow2')
        if self.config['ret_agg'] == 'min':
            next_returns = jnp.minimum(next_returns1, next_returns2)
        else:
            next_returns = (next_returns1 + next_returns2) / 2

        # The following returns will be bounded automatically
        returns = (jnp.expand_dims(batch['rewards'], axis=-1) +
                   self.config['discount'] * jnp.expand_dims(batch['masks'], axis=-1) * next_returns)
        noisy_returns = times * returns + (1 - times) * noises
        target_vector_field = returns - noises

        vector_field1 = self.network.select('critic_flow1')(
            noisy_returns, times, batch['observations'], batch['actions'], params=grad_params)
        vector_field2 = self.network.select('critic_flow2')(
            noisy_returns, times, batch['observations'], batch['actions'], params=grad_params)
        bcfm_loss = ((vector_field1 - target_vector_field) ** 2 +
                     (vector_field2 - target_vector_field) ** 2).mean(axis=-1)

        # DCFM loss
        noisy_next_returns1 = self.compute_flow_returns(
            noises, batch['next_observations'], next_actions, end_times=times,
            flow_network_name='target_critic_flow1')
        noisy_next_returns2 = self.compute_flow_returns(
            noises, batch['next_observations'], next_actions, end_times=times,
            flow_network_name='target_critic_flow2')
        if self.config['ret_agg'] == 'min':
            noisy_next_returns = jnp.minimum(noisy_next_returns1, noisy_next_returns2)
        else:
            noisy_next_returns = (noisy_next_returns1 + noisy_next_returns2) / 2
        noisy_returns = (
            jnp.expand_dims(batch['rewards'], axis=-1) +
            self.config['discount'] * jnp.expand_dims(batch['masks'], axis=-1) * noisy_next_returns
        )
        vector_field1 = self.network.select('critic_flow1')(
            noisy_returns, times, batch['observations'], batch['actions'], params=grad_params)
        vector_field2 = self.network.select('critic_flow2')(
            noisy_returns, times, batch['observations'], batch['actions'], params=grad_params)
        target_vector_field1 = self.network.select('target_critic_flow1')(
            noisy_next_returns, times, batch['next_observations'], next_actions)
        target_vector_field2 = self.network.select('target_critic_flow2')(
            noisy_next_returns, times, batch['next_observations'], next_actions)

        if self.config['ret_agg'] == 'min':
            target_vector_field = jnp.minimum(target_vector_field1, target_vector_field2)
        else:
            target_vector_field = (target_vector_field1 + target_vector_field2) / 2
        dcfm_loss = ((vector_field1 - target_vector_field) ** 2 +
                     (vector_field2 - target_vector_field) ** 2).mean(axis=-1)

        critic_loss = (self.config['bcfm_lambda'] * bcfm_loss + self.config['dcfm_lambda'] * dcfm_loss)
        critic_loss = (weights * critic_loss).mean()

        # For logging and confidence weights.
        q_noises = jax.random.normal(q_rng, (batch_size, 1))
        q1 = (q_noises + self.network.select('critic_flow1')(
            q_noises, jnp.zeros_like(q_noises), batch['observations'], batch['actions'])).squeeze(-1)
        q2 = (q_noises + self.network.select('critic_flow2')(
            q_noises, jnp.zeros_like(q_noises), batch['observations'], batch['actions'])).squeeze(-1)
        if self.config['clip_flow_returns']:
            q1 = jnp.clip(
                q1,
                self.config['min_reward'] / (1 - self.config['discount']),
                self.config['max_reward'] / (1 - self.config['discount']),
            )
            q2 = jnp.clip(
                q2,
                self.config['min_reward'] / (1 - self.config['discount']),
                self.config['max_reward'] / (1 - self.config['discount']),
            )
        if self.config['q_agg'] == 'min':
            q = jnp.minimum(q1, q2)
        else:
            q = (q1 + q2) / 2
        q_stds = ret_stds

        return critic_loss, {
            'critic_loss': critic_loss,
            'bcfm_loss': bcfm_loss,
            'dcfm_loss': dcfm_loss,
            'q_mean': q.mean(),
            'q_std': q_stds.mean(),
            'q_std_max': q_stds.max(),
            'q_std_min': q_stds.min(),
            'q_max': q.max(),
            'q_min': q.min(),
            'weight': weights.mean(),
        }

    def actor_loss(self, batch, grad_params, rng):
        """Compute the BC flow actor loss."""
        batch_size, action_dim = batch['actions'].shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        # BC flow loss.
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions']
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor_flow')(batch['observations'], x_t, t, params=grad_params)
        actor_loss = jnp.mean((pred - vel) ** 2)

        info = {
            'actor_loss': actor_loss,
        }

        return actor_loss, info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng
        rng, critic_rng, actor_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic_flow1')
        self.target_update(new_network, 'critic_flow2')

        return self.replace(network=new_network, rng=new_rng), info

    @partial(jax.jit, static_argnames=('flow_network_name', 'return_jac_eps_prod'))
    def compute_flow_returns(
        self,
        noises,
        observations,
        actions,
        init_times=None,
        end_times=None,
        flow_network_name='critic_flow',
        return_jac_eps_prod=False,
    ):
        """Compute returns from the return flow model using the Euler method."""
        noisy_returns = noises
        noisy_jac_eps_prod = jnp.ones_like(noises)
        if init_times is None:
            init_times = jnp.zeros((*noisy_returns.shape[:-1], 1), dtype=noisy_returns.dtype)
        if end_times is None:
            end_times = jnp.ones((*noisy_returns.shape[:-1], 1), dtype=noisy_returns.dtype)
        step_size = (end_times - init_times) / self.config['num_flow_steps']

        def func(carry, i):
            """
            carry: (noisy_goals, )
            i: current step index
            """
            (noisy_returns, noisy_jac_eps_prod) = carry

            times = i * step_size + init_times
            vector_field, jac_eps_prod = jax.jvp(
                lambda ret: self.network.select(flow_network_name)(ret, times, observations, actions),
                (noisy_returns, ),
                (noisy_jac_eps_prod, ),
            )

            new_noisy_returns = noisy_returns + step_size * vector_field
            new_noisy_jac_eps_prod = noisy_jac_eps_prod + step_size * jac_eps_prod
            if self.config['clip_flow_returns']:
                new_noisy_returns = jnp.clip(
                    new_noisy_returns,
                    self.config['min_reward'] / (1 - self.config['discount']),
                    self.config['max_reward'] / (1 - self.config['discount']),
                )

            return (new_noisy_returns, new_noisy_jac_eps_prod), None

        # Use lax.scan to do the iteration
        (noisy_returns, noisy_jac_eps_prod), _ = jax.lax.scan(
            func, (noisy_returns, noisy_jac_eps_prod), jnp.arange(self.config['num_flow_steps']))

        if return_jac_eps_prod:
            return noisy_returns, noisy_jac_eps_prod
        else:
            return noisy_returns

    @jax.jit
    def compute_flow_actions(
        self,
        noises,
        observations,
        init_times=None,
        end_times=None,
    ):
        """Compute actions from the BC flow model using the Euler method."""
        noisy_actions = noises
        if init_times is None:
            init_times = jnp.zeros((*noisy_actions.shape[:-1], 1), dtype=noisy_actions.dtype)
        if end_times is None:
            end_times = jnp.ones((*noisy_actions.shape[:-1], 1), dtype=noisy_actions.dtype)
        step_size = (end_times - init_times) / self.config['num_flow_steps']

        def func(carry, i):
            """
            carry: (noisy_goals, )
            i: current step index
            """
            (noisy_actions,) = carry

            times = i * step_size + init_times
            vector_field = self.network.select('actor_flow')(
                observations, noisy_actions, times)
            new_noisy_actions = noisy_actions + vector_field * step_size
            if self.config['clip_flow_actions']:
                new_noisy_actions = jnp.clip(new_noisy_actions, -1, 1)

            return (new_noisy_actions,), None

        # Use lax.scan to do the iteration
        (noisy_actions,), _ = jax.lax.scan(
            func, (noisy_actions,), jnp.arange(self.config['num_flow_steps']))

        if not self.config['clip_flow_actions']:
            noisy_actions = jnp.clip(noisy_actions, -1, 1)

        return noisy_actions

    @jax.jit
    def sample_actions(
        self,
        observations,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions using rejection sampling."""
        action_seed, q_seed, ret_seed = jax.random.split(seed, 3)
        actor_noises = jax.random.normal(
            action_seed,
            (*observations.shape[: -len(self.config['ob_dims'])],
             self.config['num_samples'], self.config['action_dim'])
        )
        n_observations = jnp.repeat(
            jnp.expand_dims(observations, -2),
            self.config['num_samples'],
            axis=-2,
        )
        flow_actions = self.compute_flow_actions(actor_noises, n_observations)

        q_noises = jax.random.normal(
            q_seed,
            (*observations.shape[: -len(self.config['ob_dims'])], self.config['num_samples'], 1)
        )

        q1 = (q_noises + self.network.select('critic_flow1')(
            q_noises, jnp.zeros_like(q_noises), n_observations, flow_actions)).squeeze(-1)
        q2 = (q_noises + self.network.select('critic_flow2')(
            q_noises, jnp.zeros_like(q_noises), n_observations, flow_actions)).squeeze(-1)
        if self.config['clip_flow_returns']:
            q1 = jnp.clip(
                q1,
                self.config['min_reward'] / (1 - self.config['discount']),
                self.config['max_reward'] / (1 - self.config['discount']),
            )
            q2 = jnp.clip(
                q2,
                self.config['min_reward'] / (1 - self.config['discount']),
                self.config['max_reward'] / (1 - self.config['discount']),
            )

        if self.config['q_agg'] == 'min':
            q = jnp.minimum(q1, q2)
        else:
            q = (q1 + q2) / 2

        if len(q.shape) > 1:
            actions = flow_actions[jnp.arange(q.shape[0]), jnp.argmax(q, axis=-1)]
        else:
            actions = flow_actions[jnp.argmax(q, axis=-1)]

        return actions

    @classmethod
    def create(
        cls,
        seed,
        example_batch,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            example_batch: Example batch.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_observations = example_batch['observations']
        ex_actions = example_batch['actions']
        ex_returns = ex_actions[..., :1]
        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]
        min_reward = example_batch['min_reward']
        max_reward = example_batch['max_reward']

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic_flow'] = encoder_module()
            encoders['target_critic_flow'] = encoder_module()
            encoders['actor_flow'] = encoder_module()

        # Define networks.
        critic_flow1_def = ValueVectorField(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
            num_ensembles=1,
            encoder=encoders.get('critic_flow'),
        )
        critic_flow2_def = ValueVectorField(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
            num_ensembles=1,
            encoder=encoders.get('critic_flow'),
        )
        # declare target critic explicit to prevent errors for visual tasks
        target_critic_flow1_def = ValueVectorField(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
            num_ensembles=1,
            encoder=encoders.get('target_critic_flow'),
        )
        target_critic_flow2_def = ValueVectorField(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
            num_ensembles=1,
            encoder=encoders.get('target_critic_flow'),
        )
        actor_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_flow'),
        )

        network_info = dict(
            critic_flow1=(critic_flow1_def, (ex_returns, ex_times, ex_observations, ex_actions)),
            critic_flow2=(critic_flow2_def, (ex_returns, ex_times, ex_observations, ex_actions)),
            target_critic_flow1=(target_critic_flow1_def, (ex_returns, ex_times, ex_observations, ex_actions)),
            target_critic_flow2=(target_critic_flow2_def, (ex_returns, ex_times, ex_observations, ex_actions)),
            actor_flow=(actor_flow_def, (ex_observations, ex_actions, ex_times)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params['modules_target_critic_flow1'] = params['modules_critic_flow1']
        params['modules_target_critic_flow2'] = params['modules_critic_flow2']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        config['min_reward'] = min_reward
        config['max_reward'] = max_reward
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='value_flows',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            min_reward=ml_collections.config_dict.placeholder(float),  # Minimum reward (will be set automatically).
            max_reward=ml_collections.config_dict.placeholder(float),  # Maximum reward (will be set automatically).
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            actor_layer_norm=True,  # Whether to use layer normalization for the actor.
            value_layer_norm=True,  # Whether to use layer normalization for the value and the critic.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            ret_agg='mean',  # Aggregation method for return values.
            q_agg='mean',  # Aggregation method for Q values.
            clip_flow_actions=True,  # Whether to clip the intermediate flow actions.
            clip_flow_returns=True,  # Whether to clip flow returns.
            confidence_weight_temp=10.0,  # Temperature for the confidence weights.
            dcfm_lambda=1.0,  # distributional conditional flow matching loss coefficient.
            bcfm_lambda=1.0,  # bootstrapped conditional flow matching loss coefficient.
            num_samples=16,  # Number of action samples for rejection sampling.
            num_flow_steps=10,  # Number of flow steps.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
        )
    )
    return config
