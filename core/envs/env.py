from core.utils.params import EnvParams

class Env:
    def __init__(self, env_name: str, env_params: EnvParams) -> None:
        #logging
        self.logger = env_params.logger

        self.seed = env_params.seed
        self.game = env_params.game
        
        self.logger.info(
            f"-----------------------------[ {env_name} w/ seed {self.seed} ]------------------"
        )