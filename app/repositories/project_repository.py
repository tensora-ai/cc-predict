from typing import List, Optional


class ProjectRepository:
    """Repository for accessing Project data from CosmosDB."""

    def __init__(self, cosmos_client=None):
        """Initialize the repository with a CosmosDB client."""
        self.cosmos_client = cosmos_client

    def initialize(self):
        """Initialize the repository with a CosmosDB client if not provided at construction."""
        if not self.cosmos_client:
            from app.utils.database_helper_functions import create_cosmos_db_client

            self.cosmos_client = create_cosmos_db_client("projects")

    def get_all_projects_data(self) -> List[dict]:
        """Get all raw project data from the database."""
        self.initialize()
        return list(
            self.cosmos_client.query_items(
                query="SELECT * FROM c", enable_cross_partition_query=True
            )
        )

    def get_project_data_by_id(self, project_id: str) -> Optional[dict]:
        """Get raw project data by ID."""
        self.initialize()
        projects = list(
            self.cosmos_client.query_items(
                query="SELECT * FROM c WHERE c.id = @project_id",
                parameters=[{"name": "@project_id", "value": project_id}],
                enable_cross_partition_query=True,
            )
        )
        return projects[0] if projects else None
