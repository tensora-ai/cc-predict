resource "azurerm_storage_account" "count" {
  name                     = "stcount${var.customer}${var.environment}"
  location                 = var.location
  resource_group_name      = "rg-count-${var.customer}-${var.environment}-storage"
  account_tier             = "Standard"
  account_replication_type = "RAGRS"
}

resource "azurerm_storage_container" "count_images" {
  name                  = "images"
  storage_account_id    = azurerm_storage_account.count.id
  container_access_type = "container"
}

resource "azurerm_storage_container" "count_models" {
  name                  = "models"
  storage_account_id    = azurerm_storage_account.count.id
  container_access_type = "container"
}

resource "azurerm_storage_container" "count_predictions" {
  name                  = "predictions"
  storage_account_id    = azurerm_storage_account.count.id
  container_access_type = "container"
}
