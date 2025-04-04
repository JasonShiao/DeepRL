
class BaseAgent(object):
    def __init__(self, **kwargs):
        super(BaseAgent, self).__init__(**kwargs)

    def update(self) -> dict:
        """Return a dictionary of logging information."""
        raise NotImplementedError
    
    def get_action(self, obs, tensor):
        """Return the action to take in the environment."""
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError
