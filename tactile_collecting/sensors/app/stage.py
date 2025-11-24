


class DummyStage:
    """
    SensorEnv에서 stage를 요구하므로 dummy queue 로 대체
    """
    def empty(self):
        return True

    def get(self):
        return None



