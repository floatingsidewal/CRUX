param name string
param location string
param vmSize string = 'Standard_B1s'
param adminUsername string
param adminPassword string
param osType string = 'Linux' // 'Linux' or 'Windows'
param imagePublisher string = 'Canonical'
param imageOffer string = 'Ubuntu2204'
param imageSku string = '22_04-lts-gen2'
param imageVersion string = 'latest'
param networkInterfaceId string
param tags object = {}

resource vm 'Microsoft.Compute/virtualMachines@2023-09-01' = {
  name: name
  location: location
  properties: {
    hardwareProfile: {
      vmSize: vmSize
    }
    osProfile: {
      computerName: name
      adminUsername: adminUsername
      adminPassword: adminPassword
    }
    storageProfile: {
      imageReference: {
        publisher: imagePublisher
        offer: imageOffer
        sku: imageSku
        version: imageVersion
      }
      osDisk: {
        createOption: 'FromImage'
        managedDisk: {
          storageAccountType: 'Standard_LRS'
        }
      }
    }
    networkProfile: {
      networkInterfaces: [
        {
          id: networkInterfaceId
        }
      ]
    }
  }
  tags: tags
}

output id string = vm.id
output name string = vm.name