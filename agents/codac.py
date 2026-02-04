import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, ImplicitQuantileValue


class CODACAgent(flax.struct.PyTreeNode):
    """Conservative offline distributional actor critic  (CODAC) agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def quantile_huber_loss(td_errors, taus, kappa):
        element_wise_huber_loss = jnp.where(
            jnp.abs(td_errors) <= kappa,
            0.5 * td_errors ** 2,
            kappa * (jnp.abs(td_errors) - 0.5 * kappa)
        )

        huber_loss = jnp.abs(
            taus[..., None] - (td_errors < 0).astype(jnp.float32)
        ) * element_wise_huber_loss / kappa

        huber_loss = huber_loss.sum(axis=-2).mean()

        return huber_loss

    def critic_loss(self, batch, grad_params, rng):
        """Compute the CODAC critic loss."""
        batch_size = batch['observations'].shape[0]
        rng, tau_rng, choice_rng, next_action_rng, n_next_tau_rng, next_tau_rng = jax.random.split(rng, 6)
        
        taus = jax.random.uniform(tau_rng, shape=(batch_size, self.config['num_quantiles']))
        single_tau_idxs = jax.random.randint(
            choice_rng,
            shape=(batch_size, ),
            minval=0,
            maxval=self.config['num_quantiles']
        )
        single_taus = taus[jnp.arange(batch_size), single_tau_idxs]
        quantiles = self.network.select('critic')(
            batch['observations'], batch['actions'], taus, params=grad_params)  # (num_ensembles, batch_size, num_quantiles, 1)

        next_noises = jax.random.normal(
            next_action_rng,
            (batch_size, self.config['action_dim'])
        )
        next_actions = self.network.select('actor_onestep_flow')(
            batch['next_observations'], next_noises)
        next_actions = jnp.clip(next_actions, -1, 1)

        next_taus = jax.random.uniform(next_tau_rng, shape=(batch_size, self.config['num_quantiles']))
        next_quantiles = self.network.select('target_critic')(
            batch['next_observations'], next_actions, next_taus)  # (num_ensembles, batch_size, num_quantiles, 1)
        if self.config['quantile_agg'] == 'min':
            next_quantiles = next_quantiles.min(axis=0)
        else:
            next_quantiles = next_quantiles.mean(axis=0)
        next_quantiles = next_quantiles.transpose([0, 2, 1])  # (batch_size, 1, num_quantiles)
        target_quantiles = batch['rewards'][:, None, None] + self.config['discount'] * next_quantiles

        td_errors = target_quantiles[None] - quantiles  # (num_ensembles, batch_size, num_taus, num_tau_primes)
        quantile_huber_loss = self.quantile_huber_loss(td_errors, taus, self.config['kappa'])

        # CODAC penalty
        rng, rand_action_rng, action_rng, action_log_prob_rng = jax.random.split(rng, 4)

        random_actions = jax.random.uniform(
            rand_action_rng,
            shape=(batch_size, self.config['num_logsumexp_samples'], self.config['action_dim']),
            minval=-1.0,
            maxval=1.0,
        )
        random_action_log_probs = jnp.log(0.5 ** self.config['action_dim'])

        # one-step action log probability estimation
        noises = jax.random.normal(
            action_rng,
            (batch_size, self.config['num_logsumexp_samples'], self.config['action_dim'])
        )
        log_prob_noises = jax.random.normal(
            action_log_prob_rng,
            (batch_size, self.config['num_logsumexp_samples'], self.config['action_dim'])
        )
        n_observations = jnp.repeat(
            batch['observations'][:, None],
            self.config['num_logsumexp_samples'],
            axis=1
        )

        actions, jac_dot_z = jax.jvp(
            lambda ns: self.network.select('actor_onestep_flow')(
                n_observations, ns
            ),
            (noises,),
            (log_prob_noises,)
        )
        # Hutchinson's trace estimator
        div = jnp.sum(log_prob_noises * jac_dot_z, axis=-1)

        actions = jnp.clip(actions, -1, 1)
        gaussian_log_probs = -0.5 * jnp.sum(jnp.log(2 * jnp.pi) + noises ** 2, axis=-1)
        action_log_probs = gaussian_log_probs - div

        q = jnp.mean(quantiles.squeeze(-1), axis=-1)
        observations = jnp.repeat(
            batch['observations'][:, None],
            self.config['num_logsumexp_samples'],
            axis=1
        )
        single_taus = jnp.repeat(
            single_taus[:, None, None],
            self.config['num_logsumexp_samples'],
            axis=1,
        )
        random_quantiles = self.network.select('critic')(observations, random_actions, single_taus, params=grad_params)
        quantiles = self.network.select('critic')(observations, actions, single_taus, params=grad_params)
        if self.config['quantile_agg'] == 'min':
            random_quantiles = random_quantiles.min(axis=0)
            quantiles = quantiles.min(axis=0)
        else:
            random_quantiles = random_quantiles.mean(axis=0)
            quantiles = quantiles.mean(axis=0)

        log_scores = jnp.concatenate([
            random_quantiles.squeeze() - random_action_log_probs,
            quantiles.squeeze() - action_log_probs
        ], axis=1)
        penalty_loss = jnp.mean(jax.nn.logsumexp(log_scores, axis=-1) - q)

        critic_loss = quantile_huber_loss + self.config['alpha_penalty'] * penalty_loss

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def actor_loss(self, batch, grad_params, rng):
        """Compute the one-step flow actor loss."""
        batch_size, action_dim = batch['actions'].shape
        rng, x_rng, t_rng, q_rng, actor_rng = jax.random.split(rng, 5)

        # BC flow loss.
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions']
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor_flow')(batch['observations'], x_t, t, params=grad_params)
        bc_flow_loss = jnp.mean((pred - vel) ** 2)

        batch_size = batch['observations'].shape[0]
        rng, noise_rng, tau_rng = jax.random.split(rng, 3)

        noises = jax.random.normal(noise_rng, (batch_size, action_dim))
        target_flow_actions = self.compute_flow_actions(noises, batch['observations'])
        actor_actions = self.network.select('actor_onestep_flow')(
            batch['observations'], noises, params=grad_params)
        actor_actions = jnp.clip(actor_actions, -1, 1)
        distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)

        taus = jax.random.uniform(tau_rng, shape=(batch_size, self.config['num_quantiles']))
        quantiles = self.network.select('critic')(batch['observations'], actor_actions, taus)
        if self.config['quantile_agg'] == 'min':
            quantiles = quantiles.min(axis=0)
        else:
            quantiles = quantiles.mean(axis=0)
        q = jnp.mean(quantiles.squeeze(-1), axis=-1)

        # Normalize Q values by the absolute mean to make the loss scale invariant.
        q_loss = -q.mean()
        if self.config['normalize_q_loss']:
            lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
            q_loss = lam * q_loss

        # Total loss.
        actor_loss = bc_flow_loss + self.config['alpha'] * distill_loss + q_loss

        info = {
            'actor_loss': actor_loss,
            'bc_flow_loss': bc_flow_loss,
            'distill_loss': distill_loss,
            'q_loss': q_loss,
            'q_mean': q.mean(),
            'q_abs_mean': jnp.abs(q).mean(),
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
        """Sample actions from the actor."""
        seed, action_seed = jax.random.split(seed)
        noises = jax.random.normal(
            action_seed,
            (
                *observations.shape[:-len(self.config['ob_dims'])],
                self.config['action_dim'],
            ),
        )
        actions = self.network.select('actor_onestep_flow')(observations, noises)
        actions = jnp.clip(actions, -1, 1)

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
        rng, tau_rng, init_rng = jax.random.split(rng, 3)

        ex_observations = example_batch['observations']
        ex_actions = example_batch['actions']
        ex_taus = jax.random.uniform(
            tau_rng, shape=(*ex_observations.shape[:-1], config['num_quantiles']))
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
        critic_def = ImplicitQuantileValue(
            hidden_dims=config['value_hidden_dims'],
            tau_embedding_num_cosines=config['num_cosines'],
            embedding_dim=config['embedding_dim'],
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
            critic=(critic_def, (ex_observations, ex_actions, ex_taus)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions, ex_taus)),
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

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='codac',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            value_layer_norm=True,  # Whether to use layer normalization for the value/critic.
            actor_layer_norm=True,  # Whether to use layer normalization for the actor.
            num_cosines=64,  # Number of cosines for the probability (tau) embeddings.
            embedding_dim=512,  # Intermediate embedding dimension in the IQN.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            kappa=0.9,  # Quantile regression threshold.
            quantile_agg='mean',  # Aggregation method for quantiles.
            clip_flow_actions=True,  # Whether to clip the intermediate flow actions.
            alpha=10.0,  # Distillation coefficient in DDPG+BC.
            alpha_penalty=0.1,  # Penalty coefficient.
            num_quantiles=16,  # Number of quantile samples for estimating Q.
            num_logsumexp_samples=10,  # Number of action samples for estimating logsumexp.
            num_flow_steps=10,  # Number of flow steps.
            normalize_q_loss=False,  # Whether to normalize the Q loss.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
        )
    )
    return config
