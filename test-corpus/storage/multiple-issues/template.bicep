// TEST CASE: Storage account with MULTIPLE misconfigurations
// Expected: CRITICAL risk, multiple labels
// This represents a worst-case scenario found in legacy environments

@description('Storage account with multiple security issues - KNOWN MISCONFIGURATIONS')
param location string = resourceGroup().location
param storageAccountName string = 'stlegacy${uniqueString(resourceGroup().id)}'

resource storageAccount 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: storageAccountName
  location: location
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
  properties: {
    // MISCONFIGURATION 1: Public blob access enabled
    allowBlobPublicAccess: true

    // MISCONFIGURATION 2: HTTP traffic allowed
    supportsHttpsTrafficOnly: false

    // MISCONFIGURATION 3: Weak TLS version
    minimumTlsVersion: 'TLS1_0'

    // MISCONFIGURATION 4: No soft delete (implicit - not configured)
    // Encryption is at least enabled by default
  }
}

// MISCONFIGURATION 5: No blob versioning configured
resource blobService 'Microsoft.Storage/storageAccounts/blobServices@2023-01-01' = {
  parent: storageAccount
  name: 'default'
  properties: {
    // Versioning not enabled - data recovery difficult
    isVersioningEnabled: false
    // Soft delete not enabled
    deleteRetentionPolicy: {
      enabled: false
    }
  }
}

output storageAccountId string = storageAccount.id
