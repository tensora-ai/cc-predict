data "azurerm_cosmosdb_account" "count" {
  name                = "cosno-count-${var.customer}-${var.environment}"
  resource_group_name = "rg-count-${var.customer}-${var.environment}-storage"
}

data "azurerm_cosmosdb_sql_database" "count" {
  name                = "cosmos-count-${var.customer}-${var.environment}"
  resource_group_name = "rg-count-${var.customer}-${var.environment}-storage"
  account_name        = data.azurerm_cosmosdb_account.count.name
}

resource "azurerm_cosmosdb_sql_container" "predictions_container" {
  name                  = "predictions"
  resource_group_name   = "rg-count-${var.customer}-${var.environment}-storage"
  account_name          = data.azurerm_cosmosdb_account.count.name
  database_name         = data.azurerm_cosmosdb_sql_database.count.name
  partition_key_paths   = ["/project"]
  partition_key_version = 2
}