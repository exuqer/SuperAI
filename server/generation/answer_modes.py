from enum import Enum


class ResultStatus(str, Enum):
    RETRIEVED = "RETRIEVED"
    COMPOSED = "COMPOSED"
    PREDICTED = "PREDICTED"
    INVENTED_LABEL = "INVENTED_LABEL"
    UNVERIFIED = "UNVERIFIED"


class AnswerMode(str, Enum):
    CONFIRMED = "confirmed"
    PROBABLE = "probable"
    COMPOSITE = "composite"
    COINED = "coined"
    PARTIAL = "partial"
    UNKNOWN = "unknown"
    REFUSAL = "refusal"
