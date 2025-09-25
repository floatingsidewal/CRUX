param name string
param location string
param addressPrefixes array = ['10.0.0.0/16']
param tags object = {}

resource vnet 'Microsoft.Network/virtualNetworks@2023-05-01' = {
  name: name
  location: location
  properties: {
    addressSpace: {
      addressPrefixes: addressPrefixes
    }
  }
  tags: tags
}

output id string = vnet.id
output name string = vnet.name