import logging

logger = logging.getLogger()


class Config:
    def __init__(self, max_iterations, implausibility_threshold, non_implausible_target, **kwargs):
        logger.info("Creating Config object")
        self.max_iterations = max_iterations
        self.implausibility_threshold = implausibility_threshold
        self.non_implausible_target = non_implausible_target
        self.user = {}
        self.user.update(kwargs)

        return
