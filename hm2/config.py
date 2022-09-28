import logging

logger = logging.getLogger()


class Config:
    def __init__(self, max_iterations, implausibility_threshold):
        logger.info("Creating Config object")
        self.max_iterations = max_iterations
        self.implausibility_threshold = implausibility_threshold

        return
