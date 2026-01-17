class ParallelExecutionNode:
    """
    Dummy node used to fan out execution to parallel branches.
    """

    def __call__(self, state):
        # No state modification
        return {}
