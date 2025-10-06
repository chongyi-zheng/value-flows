from typing import Any, Optional, Sequence

import distrax
import flax.linen as nn
import jax.numpy as jnp


def default_init(scale=1.0):
    """Default kernel initializer."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


def ensemblize(cls, num_qs, in_axes=None, out_axes=0, **kwargs):
    """Ensemblize a module."""
    return nn.vmap(
        cls,
        variable_axes={'params': 0, 'intermediates': 0},
        split_rngs={'params': True},
        in_axes=in_axes,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


class Identity(nn.Module):
    """Identity layer."""

    def __call__(self, x):
        return x


class MLP(nn.Module):
    """Multi-layer perceptron.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        activations: Activation function.
        activate_final: Whether to apply activation to the final layer.
        kernel_init: Kernel initializer.
        layer_norm: Whether to apply layer normalization.
    """

    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, training=False):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
                if self.dropout_rate > 0.0:
                    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)
            if i == len(self.hidden_dims) - 2:
                self.sow('intermediates', 'feature', x)
        return x


class LogParam(nn.Module):
    """Scalar parameter module with log scale."""

    init_value: float = 1.0

    @nn.compact
    def __call__(self):
        log_value = self.param('log_value', init_fn=lambda key: jnp.full((), jnp.log(self.init_value)))
        return jnp.exp(log_value)


class TransformedWithMode(distrax.Transformed):
    """Transformed distribution with mode calculation."""

    def mode(self):
        return self.bijector.forward(self.distribution.mode())


class CosineEmbedding(nn.Module):
    """Cosine embedding"""
    num_cosines: int = 64

    @nn.compact
    def __call__(self, taus):
        freqs = jnp.pi * jnp.arange(1, self.num_cosines + 1)[None, None]
        cos_embeddings = jnp.cos(freqs * taus[..., None])

        return cos_embeddings


class Actor(nn.Module):
    """Gaussian actor network.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        log_std_min: Minimum value of log standard deviation.
        log_std_max: Maximum value of log standard deviation.
        tanh_squash: Whether to squash the action with tanh.
        state_dependent_std: Whether to use state-dependent standard deviation.
        const_std: Whether to use constant standard deviation.
        final_fc_init_scale: Initial scale of the final fully-connected layer.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    log_std_min: Optional[float] = -5
    log_std_max: Optional[float] = 2
    tanh_squash: bool = False
    state_dependent_std: bool = False
    const_std: bool = True
    final_fc_init_scale: float = 1e-2
    encoder: nn.Module = None

    def setup(self):
        self.actor_net = MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)
        self.mean_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        if self.state_dependent_std:
            self.log_std_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        else:
            if not self.const_std:
                self.log_stds = self.param('log_stds', nn.initializers.zeros, (self.action_dim,))

    def __call__(
        self,
        observations,
        temperature=1.0,
    ):
        """Return action distributions.

        Args:
            observations: Observations.
            temperature: Scaling factor for the standard deviation.
        """
        if self.encoder is not None:
            inputs = self.encoder(observations)
        else:
            inputs = observations
        outputs = self.actor_net(inputs)

        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        else:
            if self.const_std:
                log_stds = jnp.zeros_like(means)
            else:
                log_stds = self.log_stds

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)
        if self.tanh_squash:
            distribution = TransformedWithMode(distribution, distrax.Block(distrax.Tanh(), ndims=1))

        return distribution


class Value(nn.Module):
    """Value/critic network.

    This module can be used for both value V(s, g) and critic Q(s, a, g) functions.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        value_dim: Value dimension.
        layer_norm: Whether to apply layer normalization.
        num_ensembles: Number of ensemble components.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    value_dim: int = 1
    layer_norm: bool = True
    num_ensembles: int = 2
    encoder: nn.Module = None

    def setup(self):
        mlp_class = MLP
        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class, self.num_ensembles)
        value_net = mlp_class((*self.hidden_dims, self.value_dim), activate_final=False, layer_norm=self.layer_norm)

        self.value_net = value_net

    def __call__(self, observations, actions=None):
        """Return values or critic values.

        Args:
            observations: Observations.
            actions: Actions (optional).
        """
        if self.encoder is not None:
            inputs = [self.encoder(observations)]
        else:
            inputs = [observations]
        if actions is not None:
            inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)

        v = self.value_net(inputs)
        if self.value_dim == 1:
            v = v.squeeze(-1)

        return v


class ValueVectorField(nn.Module):
    """Value vector field network for flow matching.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        value_dim: Value dimension.
        layer_norm: Whether to apply layer normalization.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    value_dim: int = 1
    layer_norm: bool = False
    dropout_rate: float = 0.0
    num_ensembles: int = 2
    encoder: nn.Module = None

    def setup(self) -> None:
        mlp_class = MLP
        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class, self.num_ensembles)
        value_net = mlp_class((*self.hidden_dims, 1), activate_final=False,
                              layer_norm=self.layer_norm, dropout_rate=self.dropout_rate)

        self.value_net = value_net

    @nn.compact
    def __call__(self, returns, times, observations, actions=None, is_encoded=False, training=False):
        """Return the vectors at the given states, actions, and times.

        Args:
            returns: Returns.
            times: Times.
            observations: Observations.
            actions: Actions.
            is_encoded: Whether the observations are already encoded.
            training: Whether the network is in training mode for dropout.
        """
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)
        if actions is None:
            inputs = jnp.concatenate([returns, times, observations], axis=-1)
        else:
            inputs = jnp.concatenate([returns, times, observations, actions], axis=-1)

        v = self.value_net(inputs, training=training)
        # if self.value_dim == 1:
        #     v = v.squeeze(-1)

        return v


class ActorVectorField(nn.Module):
    """Actor vector field network for flow matching.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    encoder: nn.Module = None

    def setup(self) -> None:
        self.mlp = MLP((*self.hidden_dims, self.action_dim), activate_final=False, layer_norm=self.layer_norm)

    @nn.compact
    def __call__(self, observations, actions, times=None, is_encoded=False):
        """Return the vectors at the given states, actions, and times (optional).

        Args:
            observations: Observations.
            actions: Actions.
            times: Times (optional).
            is_encoded: Whether the observations are already encoded.
        """
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)
        if times is None:
            inputs = jnp.concatenate([observations, actions], axis=-1)
        else:
            inputs = jnp.concatenate([observations, actions, times], axis=-1)

        v = self.mlp(inputs)

        return v


class ImplicitQuantileValue(nn.Module):
    """implicit quantile value/critic network.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        tau_embedding_num_cosines: Number of cosines in the tau embedding.
        embedding_dim: Embedding dimension.
        layer_norm: Whether to apply layer normalization.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    tau_embedding_num_cosines: int
    embedding_dim: int
    layer_norm: bool = False
    num_ensembles: int = 2
    encoder: nn.Module = None

    def setup(self) -> None:
        mlp_class = MLP
        self.tau_cosine_embedding = CosineEmbedding(
            num_cosines=self.tau_embedding_num_cosines)
        self.tau_embedding_net = mlp_class(
            (self.embedding_dim,), activate_final=False, layer_norm=self.layer_norm)
        self.sa_embedding_net = mlp_class(
            (self.embedding_dim,), activate_final=False, layer_norm=self.layer_norm)

        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class, self.num_ensembles)
        value_net = mlp_class((*self.hidden_dims, 1), activate_final=False, layer_norm=self.layer_norm)
        self.value_net = value_net

    @nn.compact
    def __call__(self, observations, actions, taus):
        if self.encoder is not None:
            observations = self.encoder(observations)
        tau_cosine_embeddings = self.tau_cosine_embedding(taus)
        tau_embeddings = self.tau_embedding_net(tau_cosine_embeddings)
        sa_embeddings = self.sa_embedding_net(
            jnp.concatenate([observations, actions], axis=-1))
        embeddings = (jnp.expand_dims(sa_embeddings, axis=-2) * tau_embeddings)  # (num_ensembles, batch_size, ..., num_taus, embedding_dims)
        quantiles = self.value_net(embeddings)  # (num_ensembles, batch_size, ..., num_taus, 1)
        
        return quantiles
