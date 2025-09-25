param name string
param location string
@description('Array of NSG rule objects with fields: name, priority, direction, access, protocol, sourcePortRange, destinationPortRange, sourceAddressPrefix, destinationAddressPrefix')
param rules array = [
  {
    name: 'AllowHttp'
    priority: 100
    direction: 'Inbound'
    access: 'Allow'
    protocol: '*'
    sourcePortRange: '*'
    destinationPortRange: '80'
    sourceAddressPrefix: '*'
    destinationAddressPrefix: '*'
  }
]
param tags object = {}

resource nsg 'Microsoft.Network/networkSecurityGroups@2023-05-01' = {
  name: name
  location: location
  properties: {
    securityRules: [for r in rules: {
      name: r.name
      properties: {
        priority: r.priority
        direction: r.direction
        access: r.access
        protocol: r.protocol
        sourcePortRange: r.sourcePortRange
        destinationPortRange: r.destinationPortRange
        sourceAddressPrefix: r.sourceAddressPrefix
        destinationAddressPrefix: r.destinationAddressPrefix
      }
    }]
  }
  tags: tags
}

output id string = nsg.id
output name string = nsg.name

