import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value


class C51Agent(flax.struct.PyTreeNode):
    """Categorical deep q-learning (C51) agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        """Compute the C51 critic loss."""
        batch_size = batch['observations'].shape[0]
        rng, next_action_rng = jax.random.split(rng)

        offset = jnp.arange(batch_size, dtype=jnp.int32) * self.config['num_atoms']
        offset = offset[:, None]
        offset = jnp.broadcast_to(offset, (batch_size, self.config['num_atoms']))

        logits = self.network.select('critic')(
            batch['observations'], batch['actions'], params=grad_params)

        next_actions = self.sample_actions(batch['next_observations'], next_action_rng)

        next_logits = self.network.select('target_critic')(
            batch['next_observations'], next_actions)
        next_probs = jax.nn.softmax(next_logits, axis=-1)
        next_probs = next_probs.mean(axis=0)

        projected_atoms = jnp.clip(
            batch['rewards'][:, None] + self.config['discount'] * batch['masks'][:, None] * self.config['atoms'][None],
            self.config['v_min'],
            self.config['v_max'],
        )
        projected_bins = (projected_atoms - self.config['v_min']) / self.config['delta_atom']
        lower_bins = jnp.floor(projected_bins).astype(jnp.int32)
        upper_bins = jnp.ceil(projected_bins).astype(jnp.int32)

        delta_mass_lower = (upper_bins + (lower_bins == upper_bins).astype(jnp.int32) - projected_bins) * next_probs
        delta_mass_upper = (projected_bins - lower_bins) * next_probs

        mass = jnp.zeros(batch_size * self.config['num_atoms'])
        mass = mass.at[(lower_bins + offset).ravel()].add(delta_mass_lower.ravel())
        mass = mass.at[(upper_bins + offset).ravel()].add(delta_mass_upper.ravel())
        mass = mass.reshape(batch_size, self.config['num_atoms'])

        critic_loss = jnp.sum(-mass[None] * logits, axis=-1).mean()

        # for logging
        probs = jax.nn.softmax(logits, axis=-1)
        q = jnp.sum(probs * self.config['atoms'][None, None], axis=-1)

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def actor_loss(self, batch, grad_params, rng):
        """Compute the BC flow actor loss."""
        batch_size, action_dim = batch['actions'].shape
        rng, x_rng, t_rng, q_rng, actor_rng = jax.random.split(rng, 5)

        # BC flow loss.
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions']
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor_flow')(
            batch['observations'], x_t, t, params=grad_params)
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
        self.target_update(new_network, 'critic')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def compute_flow_actions(
        self,
        noises,
        observations,
        init_times=None,
        end_times=None,
    ):
        """Sample actions from the flow actor using the Euler method."""
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
        n_noises = jax.random.normal(
            seed,
            (*observations.shape[:-len(self.config['ob_dims'])],
             self.config['num_samples'],
             self.config['action_dim'])
        )
        n_observations = jnp.repeat(
            jnp.expand_dims(observations, -2),
            self.config['num_samples'],
            axis=-2,
        )
        n_actions = self.compute_flow_actions(n_noises, n_observations)

        n_logits = self.network.select('critic')(n_observations, n_actions)
        n_probs = jax.nn.softmax(n_logits, axis=-1)
        qs = jnp.sum(n_probs * self.config['atoms'][None, None], axis=-1)
        if self.config['q_agg'] == 'min':
            q = qs.min(axis=0)
        else:
            q = qs.mean(axis=0)

        actions = n_actions[jnp.argmax(q)]

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
        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor'] = encoder_module()

        # Define networks.
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            value_dim=config['num_atoms'],
            layer_norm=config['value_layer_norm'],
            num_ensembles=2,
            encoder=encoders.get('critic'),
        )
        actor_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_flow'),
        )
        actor_onestep_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_onestep_flow'),
        )

        network_info = dict(
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            actor_flow=(actor_flow_def, (ex_observations, ex_actions, ex_times)),
            actor_onestep_flow=(actor_onestep_flow_def, (ex_observations, ex_actions)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params['modules_target_critic'] = params['modules_critic']

        config['v_min'] = example_batch['min_reward'] / (1 - config['discount'])
        config['v_max'] = example_batch['max_reward'] / (1 - config['discount'])

        atoms = jnp.linspace(config['v_min'], config['v_max'], config['num_atoms'])
        delta_atom = (config['v_max'] - config['v_min']) / (config['num_atoms'] - 1)
        config['atoms'] = atoms
        config['delta_atom'] = delta_atom

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='c51',  # Agent name.
            v_min=ml_collections.config_dict.placeholder(float),  # Minimum value of the z support (will be set automatically).
            v_max=ml_collections.config_dict.placeholder(float),  # Maximum value of the z support (will be set automatically).
            atoms=ml_collections.config_dict.placeholder(jnp.ndarray),  # Value of each atom of the z support (will be set automatically).
            delta_atom=ml_collections.config_dict.placeholder(float),  # Delta value between each atom (will be set automatically).
            ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            value_layer_norm=True,  # Whether to use layer normalization for the value/critic.
            actor_layer_norm=True,  # Whether to use layer normalization for the actor.
            num_atoms=51,  # Number of atoms in the z support.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            q_agg='mean',  # Aggregation method for quantiles.
            clip_flow_actions=True,  # Whether to clip the intermediate flow actions.
            num_samples=16,  # Number of action samples for rejection sampling.
            num_flow_steps=10,  # Number of flow steps.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
        )
    )
    return config
