from decomposition_core import BackpropMLP
from LocalBPCNNBenchmark import BPCNN, config_from_args, parse_args
from MLPMixerBenchmarkSuite import BackpropMixer


def test_all_terminal_bp_classifiers_are_bias_free() -> None:
    mlp = BackpropMLP(16, 3, hidden_dims=(8,))
    cnn = BPCNN(3, 10)
    mixer = BackpropMixer(3, 10, 32, 4, 16, 2, 16, 64)
    assert mlp.classifier.bias is None
    assert cnn.head.classifier.bias is None
    assert mixer.classifier.bias is None


def test_cnn_configuration_records_bias_ablation() -> None:
    config = config_from_args(parse_args(["mnist", "--method", "bp", "--device", "cpu"]))
    assert config.terminal_classifier_bias is False
