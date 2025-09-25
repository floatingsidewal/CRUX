param name string
param location string
param skuName string = 'Basic' // 'Basic' or 'Standard'
param allocationMethod string = 'Dynamic' // 'Static' or 'Dynamic'
param tags object = {}

resource pip 'Microsoft.Network/publicIPAddresses@2023-05-01' = {
  name: name
  location: location
  sku: {
    name: skuName
  }
  properties: {
    publicIPAllocationMethod: allocationMethod
  }
  tags: tags
}

output id string = pip.id
output name string = pip.name
output ipAddress string = pip.properties.ipAddress