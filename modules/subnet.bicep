param name string
param virtualNetworkName string
param addressPrefix string = '10.0.0.0/24'
param tags object = {}

resource subnet 'Microsoft.Network/virtualNetworks/subnets@2023-05-01' = {
  name: '${virtualNetworkName}/${name}'
  properties: {
    addressPrefix: addressPrefix
  }
}

output id string = subnet.id
output name string = subnet.name