param name string
param location string
param vmSize string = 'Standard_B1s'
param capacity int = 1
param adminUsername string
param adminPassword string
param osType string = 'Linux' // 'Linux' or 'Windows'
param imagePublisher string = 'Canonical'
param imageOffer string = 'Ubuntu2204'
param imageSku string = '22_04-lts-gen2'
param imageVersion string = 'latest'
param subnetId string
param tags object = {}

resource vmss 'Microsoft.Compute/virtualMachineScaleSets@2023-09-01' = {
  name: name
  location: location
  sku: {
    name: vmSize
    tier: 'Standard'
    capacity: capacity
  }
  properties: {
    upgradePolicy: {
      mode: 'Manual'
    }
    virtualMachineProfile: {
      osProfile: {
        computerNamePrefix: name
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
        networkInterfaceConfigurations: [
          {
            name: '${name}-nic'
            properties: {
              primary: true
              ipConfigurations: [
                {
                  name: '${name}-ipconfig'
                  properties: {
                    subnet: {
                      id: subnetId
                    }
                  }
                }
              ]
            }
          }
        ]
      }
    }
  }
  tags: tags
}

output id string = vmss.id
output name string = vmss.name