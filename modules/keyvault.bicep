param name string
param location string
param skuName string = 'standard' // 'standard' or 'premium'
param enablePurgeProtection bool = true
param publicNetworkAccess string = 'Enabled' // 'Enabled' or 'Disabled'
param defaultAction string = 'Deny' // network ACL default action
param tenantId string = subscription().tenantId
param tags object = {}

resource kv 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: name
  location: location
  properties: {
    tenantId: tenantId
    enablePurgeProtection: enablePurgeProtection
    enableRbacAuthorization: true
    publicNetworkAccess: publicNetworkAccess
    networkAcls: {
      defaultAction: defaultAction
      bypass: 'AzureServices'
    }
    sku: {
      family: 'A'
      name: skuName
    }
  }
  tags: tags
}

output id string = kv.id
output name string = kv.name

