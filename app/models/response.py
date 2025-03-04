from pydantic import BaseModel


class HealthCheckResponse(BaseModel):
    status: str


class PredictReturnParams(BaseModel, extra="forbid"):
    id: str
    camera: str
    position: str
    project: str
    timestamp: str
    counts: dict[str, int]

    def to_cosmosdb_entry(self) -> dict:
        return self.model_dump()
