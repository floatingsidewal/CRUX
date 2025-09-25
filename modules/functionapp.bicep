param name string
param location string
param httpsOnly bool = true
param kind string = 'functionapp,linux'
param appSettings object = {}
param tags object = {}

resource plan 'Microsoft.Web/serverfarms@2022-03-01' = {
  name: '${name}-plan'
  location: location
  sku: {
    name: 'Y1'
    tier: 'Dynamic'
  }
  kind: 'functionapp'
  properties: {
    reserved: true // Linux
  }
  tags: tags
}

resource site 'Microsoft.Web/sites@2022-03-01' = {
  name: name
  location: location
  kind: kind
  properties: {
    serverFarmId: plan.id
    httpsOnly: httpsOnly
    siteConfig: {
      ftpsState: 'FtpsOnly'
    }
  }
  tags: tags
}

// Optional app settings
resource appsettings 'Microsoft.Web/sites/config@2022-03-01' = if (length(keys(appSettings)) > 0) {
  name: '${name}/appsettings'
  properties: appSettings
  dependsOn: [ site ]
}

output id string = site.id
output name string = site.name

