from ..exceptions import PinkError


class NoPositionLimitProvided(PinkError):
    """If neither minimum nor maximum position limits are provided."""
