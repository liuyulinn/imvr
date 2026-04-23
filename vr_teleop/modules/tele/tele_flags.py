from enum import Enum


class TeleFlags(Enum):
    IK_SUCCESS = 1

    IK_FAILED = -1
    NO_HAND_POSE = -2

    NOT_IN_CONTROL = 0

    RESET = -5
    DONE = 2

    CLEAR_ERROR = -3
