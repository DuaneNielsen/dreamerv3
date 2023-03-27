from rssm import ModernDecoder, Decoder, RSSM, SequenceModel, Encoder, DynamicsPredictor, RewardPredictor, \
    ContinuePredictor, LinearEncoder, LinearDynamicsPredictor, Embedder, EmbedderPrepro, DecoderProcessed
from actor_critic import Actor, Critic
import symlog


def prepro_symlog(observation):
    return symlog.symlog(observation.float()).permute(0, 1, 4, 2, 3) / 255.


def inv_prepro_symlog(observation_enc):
    if len(observation_enc.shape) == 4:
        return symlog.symexp(observation_enc).permute(0, 2, 3, 1) * 255.
    return symlog.symexp(observation_enc).permute(0, 1, 3, 4, 2) * 255.


def prepro(observation):
    return observation.float().permute(0, 1, 4, 2, 3) / 255.


def inv_prepro(observation_enc):
    if len(observation_enc.shape) == 4:
        return observation_enc.permute(0, 2, 3, 1) * 255.
    return observation_enc.permute(0, 1, 3, 4, 2) * 255.


class PrePostProcessing:
    def __init__(self, normalization=False):
        self.mean = None
        self.std = None
        self.target_samples = 10000
        self.samples = 0
        self.normalization=normalization

    def prepro(self, x):
        self.samples += 1
        x = x.float().permute(0, 1, 4, 2, 3) / 255.
        if self.normalization:
            if self.samples < self.target_samples:
                mean = x.mean((0, 1, -2, -1))
                std = x.std((0, 1, -2, -1))
                coeff = 1 / self.samples
                if self.mean is None:
                    self.mean = mean
                    self.std = std
                self.means = self.mean * (1 - coeff) + coeff * mean
                self.std = self.std * (1 - coeff) + coeff * std
            x = (x - self.means[None, None, :, None, None]) / self.std[None, None, :, None, None]
        return x

    def postpro(self, x):
        batch_dims = x.shape[0:-3]
        if len(batch_dims) == 2:
            if self.normalization:
                x = (x * self.std[None, None, :, None, None].to(x.device)) + self.mean[None, None, :, None, None].to(x.device)
            return x.permute(0, 1, 3, 4, 2) * 255.
        else:
            if self.normalization:
                x = (x * self.std[None, :, None, None].to(x.device)) + self.mean[None, :, None, None].to(x.device)
            return x.permute(0, 2, 3, 1) * 255.


def make(model_size, action_size, action_classes, in_channels=3, decoder=None):

    """
    As per Appendix B of the Mastering Diverse Domains through World Models paper
    :param action_classes:
    :return:
    """

    gru_recurrent_units = 1024
    cnn_multiplier = 48
    dense_hidden_units = 640
    mlp_layers = 3

    if model_size == 'extra_small':
        gru_recurrent_units = 256
        cnn_multiplier = 24
        dense_hidden_units = 256
        mlp_layers = 1

    elif model_size == 'small':
        gru_recurrent_units = 512
        cnn_multiplier = 32
        dense_hidden_units = 512
        mlp_layers = 2

    elif model_size == 'medium':
        gru_recurrent_units = 1024
        cnn_multiplier = 48
        dense_hidden_units = 640
        mlp_layers = 3

    decoder = '' if decoder is None else decoder
    if decoder == 'modern':
        decoder = ModernDecoder(out_channels=in_channels,  h_size=gru_recurrent_units)
    else:
        decoder = Decoder(cnn_multi=cnn_multiplier,  h_size=gru_recurrent_units, out_channels=in_channels)
    embedder = Embedder(cnn_multi=cnn_multiplier, in_channels=in_channels)

    preprocessor = PrePostProcessing(normalization=True)
    embedder = EmbedderPrepro(embedder, prepro=preprocessor.prepro)
    decoder = DecoderProcessed(decoder, prepro=preprocessor.prepro, postpro=preprocessor.postpro)

    rssm = RSSM(
        sequence_model=SequenceModel(action_classes, h_size=gru_recurrent_units),
        embedder=embedder,
        encoder=LinearEncoder(cnn_multi=cnn_multiplier, mlp_hidden=dense_hidden_units, h_size=gru_recurrent_units),
        decoder=decoder,
        dynamics_pred=DynamicsPredictor(mlp_size=dense_hidden_units, h_size=gru_recurrent_units),
        reward_pred=RewardPredictor(h_size=gru_recurrent_units, mlp_size=dense_hidden_units, mlp_layers=mlp_layers),
        continue_pred=ContinuePredictor(h_size=gru_recurrent_units, mlp_size=dense_hidden_units, mlp_layers=mlp_layers),
        h_size=gru_recurrent_units
    )

    actor = Actor(action_size=action_size, action_classes=action_classes, h_size=gru_recurrent_units, mlp_layers=mlp_layers, mlp_size=dense_hidden_units)
    critic = Critic(h_size=gru_recurrent_units, mlp_size=dense_hidden_units, mlp_layers=mlp_layers)
    return rssm, actor, critic
