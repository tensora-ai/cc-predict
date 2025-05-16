output "cosmos_db_endpoint" {
  description = "The endpoint of the Cosmos database"
  value       = data.azurerm_cosmosdb_account.count.endpoint
}

output "storage_account_name" {
  description = "The name of the storage account"
  value       = data.azurerm_storage_account.count.name

}

output "app_service_endpoint" {
  description = "The endpoint of the app service hosting the Count Predictions backend"
  value       = azurerm_linux_web_app.count_predictions.default_hostname
}
