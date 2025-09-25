param name string
param location string
param sku string = 'Standard_LRS'
param allowBlobPublicAccess bool = false
param minimumTlsVersion string = 'TLS1_2'
param defaultAction string = 'Deny' // 'Allow' or 'Deny'
param tags object = {}

resource sa 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: name
  location: location
  sku: {
    name: sku
  }
  kind: 'StorageV2'
  properties: {
    allowBlobPublicAccess: allowBlobPublicAccess
    minimumTlsVersion: minimumTlsVersion
    networkAcls: {
      defaultAction: defaultAction
      bypass: 'AzureServices'
    }
    supportsHttpsTrafficOnly: true
  }
  tags: tags
}

output id string = sa.id
output name string = sa.name

