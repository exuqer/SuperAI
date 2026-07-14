"""Training repository compatibility alias."""

from server.v2.repository import V2Repository


class TrainingRepository(V2Repository):
    pass
