// TEST CASE: Key Vault without purge protection
// Expected: HIGH risk, KeyVault_NoPurgeProtection label
// CIS Reference: 8.4 - Ensure the key vault is recoverable

@description('Key Vault without purge protection - KNOWN MISCONFIGURATION')
param location string = resourceGroup().location
param keyVaultName string = 'kv-nopurge-${uniqueString(resourceGroup().id)}'

resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: keyVaultName
  location: location
  properties: {
    sku: {
      family: 'A'
      name: 'standard'
    }
    tenantId: subscription().tenantId

    // MISCONFIGURATION: Purge protection disabled
    // If vault is deleted, secrets are permanently lost
    enablePurgeProtection: false

    // Soft delete is enabled (good) but without purge protection
    // an attacker or mistake can still permanently delete
    enableSoftDelete: true
    softDeleteRetentionInDays: 90

    // RBAC authorization (good practice)
    enableRbacAuthorization: true

    accessPolicies: []
  }
}

// Add a secret to demonstrate what's at risk
resource secret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  parent: keyVault
  name: 'database-connection-string'
  properties: {
    value: 'Server=myserver;Database=mydb;User=admin;Password=REDACTED'
  }
}

output keyVaultUri string = keyVault.properties.vaultUri
