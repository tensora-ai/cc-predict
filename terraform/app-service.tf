data "azurerm_container_registry" "count" {
  name                = "acrcount${var.customer}${var.environment}"
  resource_group_name = "rg-count-${var.customer}-${var.environment}-operations"
}

resource "azurerm_service_plan" "count_predictions" {
  name                = "asp-count-predictions-${var.customer}-${var.environment}"
  resource_group_name = "rg-count-${var.customer}-${var.environment}-apps"
  location            = var.location

  os_type  = "Linux"
  sku_name = "P0v3"

  tags = {
    project_name = "Tensora Count"
    customer     = var.customer
    environment  = var.environment
  }
}

resource "azurerm_linux_web_app" "count_predictions" {
  name                = "app-count-predictions-${var.customer}-${var.environment}-backend"
  resource_group_name = "rg-count-${var.customer}-${var.environment}-apps"
  location            = var.location
  service_plan_id     = azurerm_service_plan.count_predictions.id

  logs {
    application_logs {
      file_system_level = "Verbose"
    }
    http_logs {
      file_system {
        retention_in_days = 14
        retention_in_mb   = 100
      }
    }
  }

  site_config {
    application_stack {
      docker_image_name        = "count-${var.customer}-${var.environment}-predictions:latest"
      docker_registry_url      = "https://${data.azurerm_container_registry.count.login_server}"
      docker_registry_username = data.azurerm_container_registry.count.admin_username
      docker_registry_password = data.azurerm_container_registry.count.admin_password
    }

    always_on                         = true
    ftps_state                        = "Disabled"
    health_check_path                 = "/api/v1/"
    health_check_eviction_time_in_min = 2
  }

  app_settings = {
    WEBSITES_ENABLE_APP_SERVICE_STORAGE = "false"
    WEBSITES_CONTAINER_START_LIMIT      = 1800
    WEBSITES_PORT                       = 8000
    API_KEY                             = var.api_key
    BLOB_CONNECTION_STRING              = azurerm_storage_account.count.primary_connection_string
    COSMOS_DB_ENDPOINT                  = data.azurerm_cosmosdb_account.count.endpoint
    COSMOS_DB_PRIMARY_KEY               = data.azurerm_cosmosdb_account.count.primary_key
    COSMOS_DB_DATABASE_NAME             = data.azurerm_cosmosdb_sql_database.count.name
    STANDARD_MODEL_NAME                 = var.standard_model_name
    LIGHTSHOW_MODEL_NAME                = var.lightshow_model_name
  }

  lifecycle {
    ignore_changes = [
      app_settings["DOCKER_REGISTRY_SERVER_PASSWORD"],
      site_config[0].application_stack[0].docker_registry_password
    ]
  }

  tags = {
    project_name = "Tensora Count"
    customer     = var.customer
    environment  = var.environment
  }
}
