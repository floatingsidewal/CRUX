// TEST CASE: Storage account with public blob access enabled
// Expected: HIGH risk, Storage_PublicAccess label
// CIS Reference: 3.7 - Ensure 'Public access level' is set to Private for blob containers

@description('Storage account with public blob access - KNOWN MISCONFIGURATION')
param location string = resourceGroup().location
param storageAccountName string = 'stpublicaccess${uniqueString(resourceGroup().id)}'

resource storageAccount 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: storageAccountName
  location: location
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
  properties: {
    // MISCONFIGURATION: Public blob access is enabled
    allowBlobPublicAccess: true

    // Other settings are secure
    supportsHttpsTrafficOnly: true
    minimumTlsVersion: 'TLS1_2'
    encryption: {
      services: {
        blob: {
          enabled: true
        }
        file: {
          enabled: true
        }
      }
      keySource: 'Microsoft.Storage'
    }
  }
}

output storageAccountId string = storageAccount.id
