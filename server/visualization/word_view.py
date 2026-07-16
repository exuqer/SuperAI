from .views import VisualizationSuite


def build(state):
    return VisualizationSuite().space_view(state, "word_space")
